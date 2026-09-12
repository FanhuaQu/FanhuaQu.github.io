---
title: Cute Tile Copy
date: 2026-08-24 20:00:00
tags: [cute, Layout, Tensor, Copy]
categories: [Cute 学习笔记]
description: 文章介绍了 cute Tiled Copy
---

# 概述
**`TiledCopy` = 一条 `Copy_Atom` 指令 ＋ TV layout + tiler，平铺成"整个线程块协作搬运一整块数据"的引擎。**
用 `tiled_copy.get_thread_slice(threadIdx.x)` 取出**某个线程**的那份搬运切片，再用 `.partition_S(dst)` / `.partition_D(src)` 切出源/目标，最后每条 copy 指令发一个搬运。

它是 **TiledMMA 的"搬运版本"**——两者共享同一套 TV layout 机制，只是从"算"变成"搬"。


# 整体架构
Cute 对 Copy的抽象采用了五层金字塔结构，自底向上依次为
* Copy Operation: PTX封装指令：SRegisters、DRegisters、copy()
* Copy_Traits: bit级布局：ThrID、SrcLayout、DstLayout、RefLayout
* Copy_Atom: 可调用的原子Copy: call(), with()
* TiledCopy: 平铺后的Copy: tidfrg_S/D, get_layoutS/D_TV
* ThrCopy: 某个线程的视角，partition_S/D, retile_S/D
源文件分布如下:  
-------------------------------
|      层级      |              源文件               |            职责             |
| :------------: | :-------------------------------: | :-------------------------: |
| Copy Operation |    include/cute/arch/copy*.hpp    |       封装PTX内联汇编       |
|  Copy_Traits   | include/cute/atom/copy_traits.hpp |     描述T-V到bit的映射      |
|   Copy_Atom    |  include/cute/atom/copy_atom.hpp  | 提供调用接口和ValType参数化 |
|   TiledCopy    |  include/cute/atom/copy_atom.hpp  |     将Atom在MN方向平铺      |
|    ThrCopy     |  include/cute/atom/copy_atom.hpp  |   单线程的分区与执行视角    |
---------------------------------------

下面挨个看

# Copy Operation
是PTX指令的C++封装

## 基本结构
Copy Operation是最底层的抽象，直接封装PTX指令，每个Operation结构体包含：
* SRegisters：源寄存器数组类型
* DRegisters：目标寄存器数组类型
* Copy()：静态成员函数，执行数据搬运


## UniversalCopy
对于普通的标量拷贝。Cute提供了`UniversalCopy` (`include/cute/arch/copy.hpp`)


```c++
//
// Direct Copy for any specific types
//
template <class S, class D = S>
struct UniversalCopy
{
  using SRegisters = S[1];  // 一个源寄存器
  using DRegisters = D[1];  // 一个目标寄存器

  // Sanity
  static_assert(sizeof_bits_v<S> >= 8);
  static_assert(sizeof_bits_v<D> >= 8);

  CUTE_HOST_DEVICE static constexpr void
  copy(S const& src,
       D      & dst)
  {
    dst = src;  // 直接赋值
  }
};

// 自动向量化拷贝(假设最大128bit对齐)
using AutoVectorizingCopy = AutoVectorizingCopyWithAssumedAlignment<128>;
// 默认拷贝（不假设对齐）
using DefaultCopy = AutoVectorizingCopyWithAssumedAlignment<8>;
```
`AutoVectorizingCopyWithAssumedAlignment<MaxVecBits>`是一个特殊的展位类型，本身不执行具体指令，而是定义好了一个“自动向量宽度”的copy策略：“copy() 用 gcd() 在"src/dst 公共可向量元素数"与"静态对齐位宽"之间求最大安全宽度,recast 成 uint_bit_t<vec_bits> 一次搬尽量多;假设对齐的程度 = 静态则 128b、有动态则退 8b。它是个 UniversalCopy 派生 + 自适应前端的组合,也是日常裸 copy(A,B) 的默认引擎”。在实际使用过程中，Cute会根据张量的实际布局，自动将这种“通用拷贝”展开为合适宽度的`UniversalCopy<uint128_t>`等指令

## cp.async (SM80)
Ampere架构引入了`cp.async`架构，可以直接从全局数据拷贝数据到共享内存，这个拷贝是异步的，并且不需要经过寄存器中转
前面PTX已经介绍过了`cp.async`指令，这里不细讲了
```c++
/// Copy via cp.async with caching at all levels
template <class TS, class TD = TS>
struct SM80_CP_ASYNC_CACHEALWAYS
{
  using SRegisters = TS[1]; // 一个源值(gemm)
  using DRegisters = TD[1]; // 一个目标值(smem)

  static_assert(sizeof(TS) == sizeof(TD), "cp.async requires sizeof(src_value_type) == sizeof(dst_value_type)");
  static_assert(sizeof(TS) == 4 || sizeof(TS) == 8 || sizeof(TS) == 16, "cp.async sizeof(TS) is not supported");

  CUTE_HOST_DEVICE static void
  copy(TS const& gmem_src,
       TD      & smem_dst)
  {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    TS const* gmem_ptr    = &gmem_src;
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_dst);
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
        :: "r"(smem_int_ptr),   // 共享内存地址
           "l"(gmem_ptr),       // 全局内存地址
           "n"(sizeof(TS)));    // 拷贝字节数，是编译器常量
#else
    CUTE_INVALID_CONTROL_PATH("Support for cp.async instructions has not been enabled");
#endif
  }
};

/// Copy via cp.async with caching at global level
// 仅16字节
template <class TS, class TD = TS>
struct SM80_CP_ASYNC_CACHEGLOBAL{};

// 带谓词的版本

/// Copy via cp.async with caching at all levels
template <class TS, class TD = TS>
struct SM80_CP_ASYNC_CACHEALWAYS_ZFILL
{
  using SRegisters = TS[1];
  using DRegisters = TD[1];

  static_assert(sizeof(TS) == sizeof(TD), "cp.async requires sizeof(src_value_type) == sizeof(dst_value_type)");
  static_assert(sizeof(TS) == 4 || sizeof(TS) == 8 || sizeof(TS) == 16, "cp.async sizeof(TS) is not supported");

  CUTE_HOST_DEVICE static void
  copy(TS const& gmem_src,
       TD      & smem_dst,
       bool      pred)
  {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    TS const* gmem_ptr    = &gmem_src;
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_dst);
    int src_size = pred ? sizeof(TS) : 0;
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n"
        :: "r"(smem_int_ptr),
           "l"(gmem_ptr),
           "n"(sizeof(TS)),
           "r"(src_size));  // 谓词控制
#else
    CUTE_INVALID_CONTROL_PATH("Support for cp.async instructions has not been enabled");
#endif
  }
};

// 仅16字节
template <class TS, class TD = TS>
struct SM80_CP_ASYNC_CACHEGLOBAL_ZFILL{};
```
cp.async的关键特性：
* 异步执行：发起之后不阻塞，需要使用`cp.async.commit_group`提交，`cp.async.wait_group`同步
* 不经过寄存器：数据直接从gmem到smem，不占用寄存器
* 可选缓存策略，`.ca`缓存在L1和L2，`.cg`只缓存在L2
* 当使用`.cg`的时候，有更严格的对齐要求，需要sizeof(TS)==16
* ZFILL变体: 支持条件拷贝，predicate为false的时候用0填充。

## lamatrix (SM75)
`ldmatrix`是Turing架构引入的warp级矩阵加载指令，专门为MMA优化——按照MMA所需要的寄存器分布从共享内存加载数据。

```c++
// 一个warp加载一个8x8矩阵
struct SM75_U32x1_LDSM_N
{
  using SRegisters = uint128_t[1];  // 1 个 128-bit 源(smem地址)
  using DRegisters = uint32_t[1];   // 1 个

  CUTE_HOST_DEVICE static void
  copy(uint128_t const& smem_src,
       uint32_t& dst)
  {
#if defined(CUTE_ARCH_LDSM_SM75_ACTIVATED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
    asm volatile ("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"
        : "=r"(dst)
        :  "r"(smem_int_ptr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use ldmatrix without CUTE_ARCH_LDSM_SM75_ACTIVATED.");
#endif
  }
};

// x2变体：加载2个8x8矩阵
struct SM75_U32x2_LDSM_N
{
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  copy(uint128_t const& smem_src,
       uint32_t& dst0, uint32_t& dst1)
  {
#if defined(CUTE_ARCH_LDSM_SM75_ACTIVATED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
    asm volatile ("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(dst0), "=r"(dst1)
        :  "r"(smem_int_ptr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use ldmatrix without CUTE_ARCH_LDSM_SM75_ACTIVATED.");
#endif
  }
};
```
在PTX的笔记里详细整理了ldmatrix的使用，这里不细说了
lamatrix的关键特性：
* warp级别，32个thread协作，每个线程提供一个smem地址，硬件按照8x8矩阵布局将数据分发到各个线程的寄存器
* x1/x2/x4，支持加载1/2/4个8x8矩阵(详见PTX章节)
* `.N` vs `.T`控制加载时的转置行为
* 与MMA对齐：正好匹配了`mma.sync`指令对A/B矩阵的寄存器分布要求

## TMA (SM90)
Hopper架构引入了TMA(Tensor Memory Accelerator)，这是对Copy操作抽象的最大变革。`cp.async`引入了异步拷贝，但是还是需要线程发射指令，会占用指令发射槽。TMA则引入了专用的硬件，自动处理地址计算、边界检查和swizzle
`include/cute/arch/copy_sm90_tma.hpp`
```c++
struct SM90_TMA_LOAD_1D
{
  CUTE_HOST_DEVICE static void
  copy( void const* desc_ptr,   // TMA描述符
        uint64_t* mbar_ptr,     // mbarrier指针
        uint64_t cache_hint,    // L2缓存提示
        void      * smem_ptr,   // smem目标地址
        int32_t const& crd0)    // 坐标
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_load(__LINE__, gmem_int_desc, smem_int_mbar, smem_int_ptr);
#if defined(CUTE_ARCH_TMA_SM120_ENABLED)
    asm volatile (
      "cp.async.bulk.tensor.1d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3}], [%2], %4;"
      :
      : "r"(smem_int_ptr),  // smem目标
        "l"(gmem_int_desc), // TMA描述符
        "r"(smem_int_mbar), // mbarrier
        "r"(crd0),          // 张量坐标
        "l"(cache_hint)     // 缓存提示
      : "memory");
#else
    asm volatile (
      "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3}], [%2], %4;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "l"(cache_hint)
      : "memory");
#endif
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }

// 此外还支持prefetch到L2
  struct PREFETCH
  {
    CUTE_HOST_DEVICE static void
    copy(void const* desc_ptr,
         int32_t const& crd0)
    {
  #if defined(CUTE_ARCH_TMA_SM90_ENABLED)
      uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
      asm volatile (
        "cp.async.bulk.prefetch.tensor.1d.L2.global"
        " [%0, {%1}];"
        :
        : "l"(gmem_int_desc),
          "r"(crd0)
        : "memory");
  #else
      CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
  #endif
    }
  };
};
```
TMA相关特性：
* 描述符驱动：通过`TmaDescriptor`(TensorMap)描述全局内存中的张量形状、步长和swizzle模式，硬件会自动计算地址
* 多维支持：支持1~5维变体
* mbarrier同步：借助mbarrier和`complete_tx::bytes`机制自动通知拷贝完成
* 边界安全：硬件自动处理越界访问(OOB)，不需要像`cp.async`一样用predication
* 单线程发起：`ThrID = Layout<_1>`只需要一个线程发起TMA指令
* 多播支持：`SM90_TMA_LOAD_MULTICAST`可以将数据同时拷贝到多个SM的共享内存(cluster内)

TMA Store和Load相似，但是不需要mbarrier
```c++
struct SM90_TMA_STORE_1D
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr,
       void const* smem_ptr,
       int32_t const& crd0)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_store(__LINE__, gmem_int_desc, smem_int_ptr);
    asm volatile (
      "cp.async.bulk.tensor.1d.global.shared::cta.bulk_group [%0, {%2}], [%1];"
      :
      : "l"(gmem_int_desc), "r"(smem_int_ptr),
        "r"(crd0)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};
```

TMA Reduce (smem → gmem 带归约)
比如在split-k的gemm场景，每个CTA计算出一个partitial accumulate(C的部分和)，需要将它累加到全局输出C。就可以借助这个带有reduce的TMA操作。后面实际测一下。
```c++
struct SM90_TMA_REDUCE_ADD_1D
{
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr,  // TMA 描述符
       void const* const smem_ptr,  // smem指针
       int32_t const& crd0)         // 坐标
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_store(__LINE__, gmem_int_desc, smem_int_ptr);
    asm volatile (
      "cp.reduce.async.bulk.tensor.1d.global.shared::cta.add.bulk_group [%0, {%2}], [%1];"
      :
      : "l"(gmem_int_desc),
        "r"(smem_int_ptr),
        "r"(crd0)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};
```

Store的`.bulk_group`异步指令发完不阻塞，需要配合相应的`commit_group`和`wait_group`
```c++
// Indicate arrival of warp issuing TMA_STORE
CUTE_HOST_DEVICE static void
tma_store_arrive() {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    cutlass::arch::synclog_emit_tma_store_arrive(__LINE__);
    asm volatile("cp.async.bulk.commit_group;");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
}

// Wait until at most Count committed TMA_STOREs are pending and all prior commits are complete
template <int Count>
CUTE_HOST_DEVICE static void
tma_store_wait() {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    asm volatile(
      "cp.async.bulk.wait_group.read %0;"
      :
      : "n"(Count)
      : "memory");
    cutlass::arch::synclog_emit_tma_store_wait(__LINE__, Count);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
}
```

## Bulk Copy (Sm90)
`SM90_BULK_COPY_G2S`是Hopper的线性(非Tensor)异步bulk拷贝，用于连续的线性内存搬运，不依赖TMA描述符.
```c++
struct SM90_BULK_COPY_G2S
{
  CUTE_HOST_DEVICE static void
  copy( void const* gmem_ptr, // 源gmem地址
        uint64_t* mbar_ptr,   // mbarrier 指针
        void* smem_ptr,       // 目标smem地址
        int32_t load_bytes)   // 搬运的字节数量
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                     :
                     : "r"(smem_int_ptr), "l"(gmem_ptr), "r"(load_bytes), "r"(smem_int_mbar)
                     : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use BULK_COPY without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }

  // 预取到L2
  struct PREFETCH
  {
    CUTE_HOST_DEVICE static void
    copy(void const* gmem_ptr, int32_t load_bytes)
    {
  #if defined(CUTE_ARCH_TMA_SM90_ENABLED)
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n"
                      :
                      : "l"(gmem_ptr), "r"(load_bytes)
                      : "memory");
  #else
      CUTE_INVALID_CONTROL_PATH("Trying to use BULK_COPY without CUTE_ARCH_TMA_SM90_ENABLED.");
  #endif
    }
  };
};
```

# Copy Traits
Copy_Traits是整个Copy抽象里面最核心的“硬件契约”，回答了：
> 这条Copy指令，本质上是一个什么样的(线程，值)偏移的映射
上一章介绍的Copy Operation只知道“执行一次Copy需要哪些寄存器”，但是不知道
* 这些寄存器中的元素如何映射到源/目标张量的坐标？
* 多线程如何完成协作拷贝
* 源和目标的T-V映射不同时如何完成转换？

Copy Traits的任务就是回答这些问题————为每个Copy Operation定义了T-V到bit的映射布局

之所以用bit而不是坐标，是因为Copy需要处理各种位宽的数据，用bit方便作为通用单位统一描述。Copy Atom通过`recast_layout`将bit布局转化为实际值类型的布局

## 
Copy Traits遵循下面的思想：
```c++
/*
 * concept Copy_Traits
 * {
 *   using ThrID     =    // 逻辑线程id -> 物理线程索引
 *
 *   using SrcLayout =    // (src-tid, src-vid) -> bit
 *   using DstLayout =    // (dst-tid, dst-vid) -> bit
 *   using RefLayout =    // (ref-tid, ref-vid) -> bit
 * };

"SrcLayout/DstLayout/RefLayout 的抽象 bit 序是任意的——它只用来构造 (ref-tid,ref-vid) → (src-tid,src-vid) 和 → (dst-tid,dst-vid) 的映射。TiledCopy 里的 Layout_TV 会先把 tile 对齐到 RefLayout,再按需映射到 Src/Dst 的 (tid,vid) 表示。"
*/
```

三种布局的含义：
* SrcLayout: 描述了源数据的线程-值分布
* DstLayout：描述目标数据上的线程-值分布
* RefLayout：作为桥梁，是TiledCopy的标准表示

RefLayout的作用：当Src和Dst的线程-值分布不同时(如`ldmatrix`)，需要一个参考分布作为中间桥梁。TiledCopy 内部使用 RefLayout 坐标，然后在执行时映射到 SrcLayout 或 DstLayout

下面看前面介绍Op的Traits
## UniversalCopy 的 Traits
UniversalCopy<S,D> 就是"一个等字节、1 线程、任意类型"的保底 copy。它的布局把 (1线程, N个bit)排成一行,N = sizeof_bits(S/D)。这解释了UniversalCopy 的普适性:任何类型 S→D 只要字节匹配就能用。
```c++
template <class S, class D>
struct Copy_Traits<UniversalCopy<S,D>>
{
  // 单线程
  using ThrID = Layout<_1>;

  // (c-thr, src-val) -> bit: 1 个线程, sizeof(S) 个比特
  using SrcLayout = Layout<Shape<_1,Int<sizeof_bits<S>::value>>>;
  // (dst-thr, dst-val) -> bit: 1 个线程, sizeof(D) 个比特
  using DstLayout = Layout<Shape<_1,Int<sizeof_bits<D>::value>>>;

  // 参考布局 = 源布局
  using RefLayout = SrcLayout;
};
```
特化版本：
```c++
template <int MaxVecBits>
struct Copy_Traits<AutoVectorizingCopyWithAssumedAlignment<MaxVecBits>>
{
  // Logical thread id to thread idx (one-thread)
  using ThrID = Layout<_1>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,_1>, Stride<_0,_0>>; // // 占位,靠 copy() 算法自推
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,_1>, Stride<_0,_0>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;
};
```

## cp.async 的 Traits
cp.async 的 Traits 与 UniversalCopy 类似，因为每条指令只涉及一个线程拷贝一个值。
*"这条 cp.async 由 1 个线程操作,src 和 dst 都是 1×sizeof_bits<D>::value bit 的一段连续 bit。"*
```c++
template <class S, class D>
struct Copy_Traits<SM80_CP_ASYNC_CACHEALWAYS<S,D>>
{
  // Logical thread id to thread idx (one-thread)
  using ThrID = Layout<_1>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,Int<sizeof_bits<S>::value>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,Int<sizeof_bits<D>::value>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;
};
```
看下ZFILL变体，引入了运行时参数pred(控制是否零填充)，并重载了copy_unpack
普通的`copy_unpack(copy_traits.hpp:113)`是"拿 `CPY_Op::SRegisters + recast + explode`逐寄存器发指令，没有谓词的概念
而`cp.async`的ZFILL需要把`pred`一路传进带谓词的那条PTX指令，因此在traits中重载了`copy_unpack`，自己接管执行
```c++
template <class S, class D>
struct Copy_Traits<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<S,D>>
{
  // Logical thread id to thread idx (one-thread)
  using ThrID = Layout<_1>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,Int<sizeof_bits<S>::value>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,Int<sizeof_bits<D>::value>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // Predicate value: true = load, false = zfill
  bool pred = true;

  // Construct a zfill variant with a given predicate value
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<S,D>>
  with(bool pred) const {
    return {pred};
  }

  // Overload copy_unpack for zfill variant to pass the predicate into the op
  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr
  void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst)
  {
    static_assert(is_gmem<TS>::value, "Expected gmem source for cp.async.");
    static_assert(is_smem<TD>::value, "Expected smem destination for cp.async.");

    Tensor rS = recast<S>(src);
    Tensor rD = recast<D>(dst);

    CUTE_STATIC_ASSERT_V(size(rS) == Int<1>{},
      "In CopyAtom, src layout doesn't vectorize into registers. This src layout is incompatible with this tiled copy.");
    CUTE_STATIC_ASSERT_V(size(rD) == Int<1>{},
      "In CopyAtom, dst layout doesn't vectorize into registers. This dst layout is incompatible with this tiled copy.");

    SM80_CP_ASYNC_CACHEALWAYS_ZFILL<S,D>::copy(rS[0], rD[0], traits.pred);
  }
};
```

## ldmatrix 的 traits: 多线程布局
`ldmatrix`是warp级指令，32个线程协作，其Traits描述了复杂的T-V分布：
`ldmatrix`指令从smem读取一整条矩阵(8x128bit)到寄存器，其Traits如下
```c++
template <>
struct Copy_Traits<SM75_U32x1_LDSM_N>
{
  // 32个线程(一个warp)
  using ThrID = Layout<_32>;

  // Map from (src-thr,src-val) to bit
  // 源是共享内存，线程按照8x4排布，每个线程提供1个128-bit地址
  // Shape((8,4), 128)
  //  - 线程维度：(8,4) = 32 个线程
  //  - 值维度：128 bits = 16 bytes
  //  - 线程 0 -> bit 0, 线程 1 -> bit 128, ..., 线程 8 -> bit 0 (但 val 不同)
  using SrcLayout = Layout<Shape <Shape <  _8,_4>,_128>,
                           Stride<Stride<_128,_0>,  _1>>;
  // Map from (dst-thr,dst-val) to bit
  // 32个线程，每个线程32 bit
  using DstLayout = Layout<Shape <_32,_32>,
                           Stride<_32, _1>>;

  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
};
```
详细解释一下：
1. `ThrID = Layout<_32>` → 逻辑线程 id 是 32 个，32 个线程各出一个 uint32 寄存器
2. `SrcLayout = ((_8,_4),_128):((_128,_0),_1)`
  * tid 形状 `(_8, _4)` (拆分 32 个线程为 8 行 × 4 组),stride `(_128,_0)`
  * val 形状_128, stride _1
  也就是src 是"一个 warp 读一段 8×4×128bit 的共享区"。其中 _4 那维 stride 0(即 4 个 tid 共享同一段地址),每个 tid 给出一个 128-bit 的 smem 地址,ldmatrix 按这个地址展开 8×8
  换句话说:32 个线程其实只提供 32/4=8 个独立的 128-bit 地址,每个地址喂给 ldmatrix 读出一个 8×8 矩阵的一行 → 8 个地址构成 8 行矩阵。这正是 m8n8 的取数规律
  val = _128，就是T0~T3读取的这一行数据的宽度，对应的Stride=0,也就是一行的4个线程拿到的是相同的地址。Stride = 128，就是相邻的一组4个元素之间间隔是128bit
3. `DstLayout = (_32,_32):(_32,_1)`
  * tid 形状 _32,val 形状 _32,stride (_32,_1)
  也就是每个线程t有一段32 bit的数据
4. `RefLayout = DstLayout`
  *  参考布局取 dst 侧(因为 ldmatrix 的"规范性"体现在寄存器侧 dst;src 侧的 smem 布局更像一个"地址列表")。TiledCopy 会先把 tile 对齐到 RefLayout 即这个 (32,32) 的 dst 分布

再看下`x2`变体(加载两个8x8矩阵)：
```c++
template <>
struct Copy_Traits<SM75_U32x2_LDSM_N>
{
  // Logical thread id to thread idx (warp)
  using ThrID = Layout<_32>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <Shape < _16,_2>,_128>,
                           Stride<Stride<_128,_0>,  _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_32,Shape <_32,   _2>>,
                           Stride<_32,Stride< _1,_1024>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
};
```
这个变体下每个线程持有2个32bit寄存器
先看`SrcLayout = Layout<Shape <Shape < _16,_2>,_128>, Stride<Stride<_128,_0>,  _1>>;`
  * 寄存器排布是8x2，每两个线程负责一行128bit的数据

`using DstLayout = Layout<Shape <_32,Shape <_32,   _2>>, Stride<_32,Stride< _1,_1024>>>;`
  * 总共32个线程，每个线程持有2个uint32数据，stride = 1024，也就是8x8的堆叠


## TMA 的 Traits：描述符驱动的布局
TMA的Traits是最复杂的，但是所有triats的“布局契约”都是一样的
不管是LOAD还是STORE，开头的4行布局几乎是一样的
```c++
  using ThrID     = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;
```
上面几个Layout显得空空如也，这是因为TMA不依赖于“每个线程提供坐标”，而是用一个`TmaDescriptor`(张量描述符) + 一个块坐标。真正的几何(shape、stride、swizzle)全部封装在 `TmaDescriptor` 和 `AuxParams` 里。

```c++
// 不可执行的 SM90_TMA_LOAD（有 tma_desc，但没有 tma_mbar）
// 必须通过 .with(tma_mbar) 构造可执行版本
template <class NumBitsPerTMA, class AuxParams_>
struct Copy_Traits<SM90_TMA_LOAD, NumBitsPerTMA, AuxParams_>
{
  using ThrID     = Layout<_1>;   // 单线程发起
  using SrcLayout = Layout<Shape<_1, NumBitsPerTMA>>;  // (1, NumBits) -> bit
  using DstLayout = Layout<Shape<_1, NumBitsPerTMA>>;
  using RefLayout = SrcLayout;

  // TMA 描述符（TensorMap），存储在 Traits 中
  TmaDescriptor tma_desc_;
  using AuxParams = AuxParams_;
  AuxParams aux_params_;   // 辅助参数（步长、swizzle 等）

  // 获取 TMA 描述符
  CUTE_HOST_DEVICE constexpr
  TmaDescriptor const* get_tma_descriptor() const {
    return &tma_desc_;
  }

  // 构造可执行版本：需要 mbarrier
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM90_TMA_LOAD_OP, NumBitsPerTMA>
  with(uint64_t& tma_mbar,
       uint16_t const& multicast_mask = 0,
       TMA::CacheHintSm90 const& cache_hint = TMA::CacheHintSm90::EVICT_NORMAL) const {
    return {&tma_desc_, &tma_mbar, static_cast<uint64_t>(cache_hint)};
  }

  // 生成 TMA 坐标张量
  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  auto
  get_tma_tensor(GShape const& g_shape) const {
    static_assert(is_congruent<decltype(g_shape), decltype(aux_params_.g_stride_)>::value);
    return make_coord_tensor(make_layout(g_shape, aux_params_.g_stride_));
  }

  // 禁止直接执行：必须先调用 .with()
  template <class TS, class SLayout, class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits const&, Tensor<TS,SLayout> const&, Tensor<TD,DLayout>&) = delete;
};

```
TMA的traits采用了两阶段设计，上面的traits只负责"持有描述符 + 描述搬运的样子"，要真正执行，必须先调用`.with(tma_mbar)`拿到可执行版本的`SM90_TMA_LOAD_OP`。这是因为`mbarrier`是运行期的东西，Cute将“描述”和“执行”拆成两层，中间用`.with()`桥接

`get_tma_tensor()`：根据全局形状生成坐标张量，用于后续分区
`NumBitsPerTMA`：模板参数，表示单次TMA拷贝的bit数

可执行版本：
```c++
template <class CopyOp, class... Args>
struct TMA_LOAD_Unpack
{
  template <class TS, class SLayout, class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits<CopyOp, Args...> const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>& dst)
  {
    static_assert(is_smem<TD>::value, "SM90_TMA_LOAD requires the destination be shared memory.");

    // src 张量存储的是 TMA 坐标
    auto src_coord = src(Int<0>{});
    void* dst_ptr = cute::raw_pointer_cast(dst.data());

    // 将参数展开并调用底层 PTX 指令
    return detail::explode_tuple(detail::CallCOPY<CopyOp>{},
                                 traits.opargs_, tuple_seq<decltype(traits.opargs_)>{},
                                 make_tuple(dst_ptr), seq<0>{},
                                 src_coord, tuple_seq<decltype(src_coord)>{});
  }
};

```

总的流程就是
不可执行版(持 desc) --with(mbar)--> 可执行版(desc+mbar+cache) --copy()--> TMA_LOAD_Unpack::copy_unpack --CallCOPY--> cp.async.bulk.tensor

# Copy Atom
`Copy_Atom`是`Copy_traits`和`元素类型`的组合，前者描述的是位级布局，而后者指明了拷贝数据的具体类型。

```c++

template <class... Args, class CopyInternalType>
struct Copy_Atom<Copy_Traits<Args...>, CopyInternalType>
  : Copy_Traits<Args...>
{
  using Traits = Copy_Traits<Args...>;

  // Bit and Thr layouts from the Copy_Traits
  using ThrID        = typename Traits::ThrID;
  using BitLayoutSrc = typename Traits::SrcLayout;
  using BitLayoutDst = typename Traits::DstLayout;
  using BitLayoutRef = typename Traits::RefLayout;

  using ValType = CopyInternalType; // 新增元素类型

  using ValLayoutSrc = decltype(recast_layout<uint1_t, ValType>(BitLayoutSrc{}));
  using ValLayoutDst = decltype(recast_layout<uint1_t, ValType>(BitLayoutDst{}));
  using ValLayoutRef = decltype(recast_layout<uint1_t, ValType>(BitLayoutRef{}));

  CUTE_STATIC_ASSERT_V(size<0>(ValLayoutSrc{}) == size(ThrID{}), "CopyOperation is not valid for Src of ValType.");
  CUTE_STATIC_ASSERT_V(size<0>(ValLayoutDst{}) == size(ThrID{}), "CopyOperation is not valid for Dst of ValType.");
  CUTE_STATIC_ASSERT_V(size<0>(ValLayoutRef{}) == size(ThrID{}), "CopyOperation is not valid for Ref of ValType.");

  static constexpr int NumValSrc = size<1>(ValLayoutSrc{});
  static constexpr int NumValDst = size<1>(ValLayoutDst{});

  // Additional Trait parameters/transformations
  template <class... TraitsArgs>
  CUTE_HOST_DEVICE
  auto
  with(TraitsArgs&&... args) const {
    auto traits = Traits::with(static_cast<TraitsArgs&&>(args)...);
    return Copy_Atom<decltype(traits), CopyInternalType>{traits};
  }

  // ......
}
```

`CopyInternalType`告诉Atom“拷贝数据的类型”，将位级布局`recast`成值级布局。
比如：
* `Copy_Atom<SM80_CP_ASYNC_CG, half_t>`，编译器会改写成`Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CG>, half_t>`,用`cp.async`拷贝128bit，解释为8个half

`Copy_Atom`继承自`Copy_Traits`，,所以它拥有`traits`的所有成员——TMA 的`atom`里你因此能拿到`tma_desc_`、`aux_params_`、调 `.with()`、调`get_tma_descriptor()`。

`Copy_Traits`的`Src/Dst/Ref`布局单位是`bit`, `Copy_Atom`把他们重铸为以`ValType`为单位的布局


## 调用接口 call()

```c++

  // Check and call instruction, or recurse
  template <class SEngine, class SLayout,
            class DEngine, class DLayout>
  CUTE_HOST_DEVICE
  void
  call(Tensor<SEngine,SLayout> const& src,
       Tensor<DEngine,DLayout>      & dst) const
  {
    static_assert(SLayout::rank == 1, "Expected rank-1 src tensor");
    static_assert(DLayout::rank == 1, "Expected rank-1 dst tensor");

    // 尺寸正好 = 一次指令的量  ——> 真正发送指令
    if constexpr (is_constant<NumValSrc, decltype(size(src))>::value ||
                  is_constant<NumValDst, decltype(size(dst))>::value) {
      // Dispatch to unpack to execute instruction
      return copy_unpack(static_cast<Traits const&>(*this), src, dst);
    } 
    // 尺寸比指令大，剥掉一层mode，递归
    else if constexpr (is_tuple<decltype(shape(src))>::value &&
                         is_tuple<decltype(shape(dst))>::value) {
      // If the size of the src/dst doesn't match the instruction,
      //   recurse this rank-1 layout by peeling off the mode
      //   ((A,B,C,...)) -> (A,B,C,...)
      return copy(*this, tensor<0>(src), tensor<0>(dst));
    } else {
      static_assert(dependent_false<SEngine>,
                    "CopyAtom: Src/Dst partitioning does not match the instruction requirement.");
    }
  }

```
当传入的张量大小和指令要求不匹配，但是形状是tuple的时候，`call`会剥掉最外层mode再递归(把大tensor逐块喂给`atom`)

还有一个带谓词的三参数版本
```c++

  // Check and call instruction, or recurse
  template <class PEngine, class PLayout,
            class SEngine, class SLayout,
            class DEngine, class DLayout>
  CUTE_HOST_DEVICE
  void
  call(Tensor<PEngine,PLayout> const& prd,
       Tensor<SEngine,SLayout> const& src,
       Tensor<DEngine,DLayout>      & dst) const
  {
    static_assert(PLayout::rank == 1, "Expected rank-1 prd tensor");
    static_assert(SLayout::rank == 1, "Expected rank-1 src tensor");
    static_assert(DLayout::rank == 1, "Expected rank-1 dst tensor");

    if constexpr (is_constant<NumValSrc, decltype(size(src))>::value ||
                  is_constant<NumValDst, decltype(size(dst))>::value) {
      // Dispatch to unpack to execute instruction
      Traits const& traits = static_cast<Traits const&>(*this);
      auto has_with_bool = cute::is_valid([](auto t)->void_t<decltype(t.with(true))>{}, traits);
      if constexpr (has_with_bool) {
        copy_unpack(traits.with(prd(Int<0>{})), src, dst);
      } else {
        if (prd(Int<0>{})) { copy_unpack(traits, src, dst); }
      }
    } else if constexpr (is_tuple<decltype(shape(prd))>::value &&
                         is_tuple<decltype(shape(src))>::value &&
                         is_tuple<decltype(shape(dst))>::value) {
      // If the size of the src/dst doesn't match the instruction,
      //   recurse this rank-1 layout by peeling off the mode
      //   ((A,B,C,...)) -> (A,B,C,...)
      return copy_if(*this, tensor<0>(prd), tensor<0>(src), tensor<0>(dst));
    } else {
      static_assert(dependent_false<SEngine>,
                    "CopyAtom: Src/Dst partitioning does not match the instruction requirement.");
    }
  }
```
逻辑是先探测traits有没有`with(bool)`，有的话就把把谓词传递给ZFILL的`with()`，没有的话，就使用if(pred)判断，不满足就跳过拷贝。


# TiledCopy
TiledCopy是Copy_Atom的"放大器"，3个模板参数的分工如下：
* Copy_Atom：一条指令 + 元素类型(一次搬运多少数据，怎么发指令)
* LayoutCopy_TV：整个tile的(thread, val) -> 坐标映射 (可以来自mma)
* ShapeTiler_MN：需要平铺覆盖的形状

`TiledCopy` 继承`Copy_Atom`,`Copy_Atom`又继承`Copy_Traits`——所以 `TiledCopy`是三层血统，既有原子指令的布局`(AtomLayout*)`,又有整块平铺的布局`(TiledLayout_TV)`


```c++

template <class Copy_Atom,
          class LayoutCopy_TV,  // (tid,vid) -> coord   [Need not be 2D...]
          class ShapeTiler_MN>  // coord space
struct TiledCopy : Copy_Atom
{
  // Layout information from the CopyAtom
  // 单次指令的TV
  using AtomThrID     = typename Copy_Atom::ThrID;        // thrid -> thr_idx
  using AtomLayoutSrc = typename Copy_Atom::ValLayoutSrc; // (thr,val) -> offset
  using AtomLayoutDst = typename Copy_Atom::ValLayoutDst; // (thr,val) -> offset
  using AtomLayoutRef = typename Copy_Atom::ValLayoutRef; // (thr,val) -> offset

  using AtomNumThr = decltype(size<0>(AtomLayoutRef{}));
  using AtomNumVal = decltype(size<1>(AtomLayoutRef{}));

  // Layout information for the TiledCopy
  // 整个tile的TV
  using Tiler_MN       = ShapeTiler_MN;
  using TiledLayout_TV = LayoutCopy_TV;
  using TiledNumThr    = decltype(size<0>(TiledLayout_TV{}));
  using TiledNumVal    = decltype(size<1>(TiledLayout_TV{}));

  CUTE_STATIC_ASSERT_V(TiledNumThr{} % AtomNumThr{} == Int<0>{}, "TiledCopy uses too few thrs for selected CopyAtom");
  CUTE_STATIC_ASSERT_V(TiledNumVal{} % AtomNumVal{} == Int<0>{}, "TiledCopy uses too few vals for selected CopyAtom");


}
```

## tidfrg_S/D + tile2thrfrg
这是`TiledCopy`的核心，将张量转换为线程-片段视图
目标是把tensor从
```
(M, N, ...)
   → zipped_divide(t, Tiler_MN) → ((TileM,TileN),(RestM,RestN,...))
   → (Thr, (FrgV,FrgX), (RestM,RestN,...))
```

* Thr: 逻辑线程号(将被 `partition_S(thr_idx)` 切掉)
* FrgV: 一次 atom 内部的值(指令内部的 mode —— demo 里 `(CopyOp, CopyM, CopyN)` 的第一维)
* FrgX: 跨 atom 平铺的值(同一个线程发的第 2、3、… 条指令)
* RestM/RestN: tile之外还有余几块

```c++
  template <class STensor>
  CUTE_HOST_DEVICE constexpr static
  auto
  tidfrg_S(STensor&& stensor)
  {
    CUTE_STATIC_ASSERT_V(rank(stensor) >= rank(Tiler_MN{}), "Rank of tensor to be partitioned too small.");

    // Tile the stensor and compute the (src-thr, src-val) -> (ref-thr, ref-val) layout
    return tile2thrfrg(zipped_divide(stensor,Tiler_MN{}), right_inverse(AtomLayoutRef{}).compose(AtomLayoutSrc{}));
  }
```

`zipped_divide(stensor, Tiler_MN)` —— 把 tensor 按 tiler 切成"tile 内容 + tile 编号", (即 `tiled_divide`)

**`right_inverse(AtomLayoutRef{}).compose(AtomLayoutSrc{})`** ← 这是 S 和 D 唯一的差别。`right_inverse(Ref)`将Ref的(thr, val)逆映射为线性索引。`.compose(Src)`再映射到Src的(thr, val)。
`right_inverse(Ref).compose(Src)` 回答的问题是:**"一个以寄存器顺序编号的值,
在源空间对应谁"**。一个 Copy_Atom 内,值可能以三种顺序出现——src 顺序、dst 顺序、ref(寄存器)顺序——而**寄存器 fragment 是以 ref 顺序存在的**。分区时以 ref 为基准排fragment 内的值序,数据却要按 src/dst 的实际地址落位,这个复合变换就完成了两套编号系统的互译。

### tile2thrfrg的四步流程
```cpp

  // Tile a tensor or a layout from shape
  //   ((TileM,TileN,...), (RestM,RestN,...))
  // to shape
  //   (Thr,(FrgV,FrgX),(RestM,RestN,...))
  template <class Tensor, class Ref2TrgLayout>
  CUTE_HOST_DEVICE constexpr static
  auto
  tile2thrfrg(Tensor&& tensor, Ref2TrgLayout const& ref2trg)
  {
    // 1、从Tiled TV 里"抠出"一个atom大小的块
    auto atom_layout_TV = zipped_divide(TiledLayout_TV{}, make_shape(AtomNumThr{}, AtomNumVal{}));
    // ((atom_tid,atom_val),(rest_tid,rest_val)) -> (m,n)

    // 2、用 ref2trg 把 atom 内部的 (tid,vid) 编号翻译成"空间中第几个 offset"
    auto trg_layout_TV = atom_layout_TV.compose(ref2trg, _);
    // ((trg_tid,trg_val),(rest_tid,rest_val)) -> (m,n)

    // 3、模式重排:把"线程两层"合并、"值两层"合并
    auto thrval2mn = coalesce(zip(trg_layout_TV), Shape<_1,Shape<_1,_1>>{});
    // ((trg_tid,rest_tid),(trg_val,rest_val)) -> (m,n)

    /// ==================

    // 4、把这个映射"复合"到 tensor 的 tile mode 上
    auto tv_tensor = tensor.compose(thrval2mn, _);
    // ((thrid,val),(RestM,RestN,...))

    // Unfold and return
    return tv_tensor(make_coord(_,_), _);
  }
```
最终结果：`(Thr, (FrgV, FrgX), (RestM, RestN, ...))`：

## retile: 重新切片
用于将已经按照某种方式切片的张量重新切片成Copy_Atom所需的布局
语义：把一个"每线程的值序列"在两种方式之间重排
* fragment视角：值按 TiledLayout_TV 里 val-major 排——方便一条条 atom 指令消费;
* tile视角：同样的值按 (M,N) 空间坐标排——方便整体访问
```c++

  // retile_S and retile_D assume they are working with the reference layout -- they are the same
  template <class Tensor>
  CUTE_HOST_DEVICE constexpr static
  auto
  retile(Tensor&& tensor)
  {
    constexpr int R = remove_cvref_t<Tensor>::rank;
    // Assert that AtomLayoutSrc|Dst is identity so we can skip the Ref transformation

    // Assume the first size<0>(tensor) elements are the first val_ids in TiledLayout_TV.
    // Then, we only need the shape+layout of those size<0>(tensor) elements in TiledLayout_TV
    //   and that shape is what we gather from the other modes of tensor

    auto V = size<0>(tensor); // 当前片段的值数

    // 第一步，找到V个值在TiledLayout_TV中的布局
    auto frg_layout_mn = upcast<TiledNumThr{} * V>(right_inverse(TiledLayout_TV{}).with_shape(shape(Tiler_MN{})));
    // (m,n) -> v_idx -- The shape and order of the V inside of TiledLayout_TV

    // 第二步，按原子值数切分
    auto frg_layout_v = zipped_divide(logical_product(make_layout(V), right_inverse(frg_layout_mn)), make_layout(AtomNumVal{}));
    // (atom_vals,rest_vals) -> (v,m,n)

    /// =======

    // 第三步：切分张量
    auto t_tensor = zipped_divide(tensor, prepend(product_each(shape(frg_layout_mn)), V));
    // ((TileV,TileM,TileN,...),(1,RestM,RestN,...))

    // Transform the tile mode
    // 应用布局
    auto v_tensor = t_tensor.compose(frg_layout_v, _);
    // ((atom_vals,rest_vals),(1,RM,RN,...))

    // Unfold and return
    return v_tensor(_, append<R>(Int<0>{},_));
  }
```
典型用途：在gemm的epilogue阶段，MMA累加器`tCrC`的值序是MMA规定的(TiledMMA的TV)，在写回gmem的时候，想用`copy(tiled_copy_C, tCrC, tCgC)`，但是TiledCopy_C的TiledLayout_TV是从 `mma.get_layoutC_TV()` 复制的,所以两者**通常直接兼容**;而 `make_tiled_copy_C_atom`(:451,子分块的流水 epilogue)做的是**把 V 截断到单 atom并重排 TV**,靠 retile 与 accumulator 对数据。

## get_layoutS_TV/get_layoutD_TV：TV 的输出接口
```cpp
static auto get_layoutS_TV() {
  // 创建参考布局
  auto ref_S = make_layout(make_shape(shape(Tiler_MN{}), Int<1>{}));

  // 通过 tile2thrfrg 转换，然后切片到 (thr_idx, val_idx) -> (M,N)
  return tile2thrfrg(ref_S, right_inverse(AtomLayoutRef{}).compose(AtomLayoutSrc{}))(_,_,Int<0>{});
}
```
作用：给一个纯"Tiler 大小的白纸 layout",用与 `tidfrg_S` 完全相同的流程排出`(thr, val) → (M,N)`。不依赖具体 tensor 也能拿到自己的 TV 布局描述。这个布局描述了哪个线程的哪个值对应源/目标张量的哪个坐标。

## get_clice / get_thread_slice: 获取单线程视角
```c++
template <class ThrIdx>
CUTE_HOST_DEVICE static
auto
get_slice(ThrIdx const& thr_idx)
{
  return ThrCopy<TiledCopy, ThrIdx>(thr_idx);
}

template <class ThrIdx>
CUTE_HOST_DEVICE static
auto
get_thread_slice(ThrIdx const& thr_idx)
{
  return get_slice(thr_idx);   // 别名
}
```

# ThrCopy
单个线程视角的分区与执行

`ThrCopy` = `tiled_copy.get_thread_slice(我的线程号)` 返回的轻量句柄,里面只存一个 `thr_idx_`,却能替你调 `TiledCopy` 的全部静态方法、并以"当前线程"的身份切出源/目标/fragment 三种视图。
```c++
template <class TiledCopy, class ThrIdx>
struct ThrCopy
{

  // 唯一的数据成员是：我的线程号
  ThrIdx thr_idx_;

  CUTE_HOST_DEVICE
  ThrCopy(ThrIdx const& thr_idx) : thr_idx_(thr_idx) {}

  // partition_S / partition_D ...
  // retile_S / retile_D ...
}
```

## partition_S / partition_D 提取本线程的片段
```c++
template <class STensor>
CUTE_HOST_DEVICE
auto
partition_S(STensor&& stensor) const {
  auto thr_tensor = make_tensor(stensor.data(), TiledCopy::tidfrg_S(stensor.layout()));
  return thr_tensor(thr_idx_, _, repeat<rank_v<STensor>>(_));
}

template <class DTensor>
CUTE_HOST_DEVICE
auto
partition_D(DTensor&& dtensor) const {
  auto thr_tensor = make_tensor(static_cast<DTensor&&>(dtensor).data(),
                                TiledCopy::tidfrg_D(dtensor.layout()));
  return thr_tensor(thr_idx_, _, repeat<rank_v<DTensor>>(_));
}
```
* `TiledCopy::tidfrg_S(stensor.layout())`: 对张量布局执行tidfrg_S，到一个`(Thr, (FrgV,FrgX), Rest…) `的布局。

## retile_S / retile_D 静态重新切片
两个关键点：
* static：是静态函数，不需要ThrCopy实例，不需要线程号，因为`retile`处理的是"这个线程自己的fragment"的值序转换，每个线程独立做，规则相同
* S/D 版**调用同一个 `TiledCopy::retile`**——`retile_S`和`retile_D`假设使用`RefLayout`工作，因为`RefLayout`通常是`SrcLayout`或`DstLayout`之一，`retile`基于`RefLayout`进行，对`Src`和`Dst`
```cpp
template <class STensor>
CUTE_HOST_DEVICE static
auto
retile_S(STensor&& stensor) {
  return make_tensor(stensor.data(), TiledCopy::retile(stensor.layout()));
}
// retile_D 同理,也调 TiledCopy::retile
```

用途：fragment 有"atom 顺序"(逐指令消费友好)与"tile 顺序"(按 (m,n)
空间访问友好)两种打包;epilogue 里把 accumulator 从 MMA 值序换到 copy 值序, 靠的就是它

# make_tiled_copy 相关函数

## make_tiled_copy: 从线程和值布局构造

最通用的工厂函数
```c++
/*
 * 从逻辑线程和值布局构造 TiledCopy。
 * 线程和值布局将坐标映射到 thr_idx 和 val_idx。
 * 取这些布局的 raked_product 生成 TV 布局和 Tiler。
 * 当线程和值需要非常特定的坐标映射时很有用。
 */
template <class... Args,
          class ThrLayout,
          class ValLayout = Layout<_1>>
CUTE_HOST_DEVICE
auto constexpr
make_tiled_copy(Copy_Atom<Args...> const& copy_atom,
                ThrLayout          const& thr_layout = {},     // (m,n) -> thr_idx
                ValLayout          const& val_layout = {})     // (m,n) -> val_idx
{
  // 第1步：raked_product 生成 (M,N) -> (thr_idx, val_idx) 的布局
  auto layout_mn = raked_product(thr_layout, val_layout);
  // 第2步：逆布局，生成 (thr_idx, val_idx) -> (M,N) 的 TV 布局
  auto layout_tv = right_inverse(layout_mn).with_shape(make_shape(size(thr_layout), size(val_layout)));
  // 第3步：生成 Tiler（提取相关元素）
  auto tiler = product_each(shape(layout_mn));

  return make_tiled_copy_impl(copy_atom, layout_tv, tiler);
}
```
* `thr_layout: (m,n) -> thr_idx`，描述哪个线程负责这个坐标
* `val_layout: (m,n) -> val_idx`, 描述这是负责者的第几个值
* `raked_product`: 将两个布局"梳状"组合，生成`(m,n) -> (thr_idx, val_idx)`
这里看一下`block_product`和`raked_producte`,设 thr=(4):(1)(4 个线程)，val=(2):(1)(每线程 2 个值)：
```bash
  blocked_product(thr, val):  thread t 拥有 {t, t+4}
        内存: [t0v0 t1v0 t2v0 t3v0 | t0v1 t1v1 t2v1 t3v1]
        → 单线程的 2 个值相隔 4,无法拼成一条 8B 访存

  raked_product(thr, val):    thread t 拥有 {2t, 2t+1}
        内存: [t0v0 t0v1 | t1v0 t1v1 | t2v0 t2v1 | t3v0 t3v1]
        → 每线程 2 个值连续,恰是一条向量指令的宽度

  区别全在 zip(get<1>, get<0>) 这一下：raked 把 val 的模式抽到最内层，使 val_layout
    的“连续”翻译成内存地址的“连续”。这就是函数体上方注释
    raked_products(:499)的用意——make_tiled_copy 生来就是为了向量化访存。教程注释
    tiled_copy.cu:203 "assumes the vector and thread layouts are aligned with contiguous
    data" 说的就是这个前提。
```

用例：
```c++
// 假如256线程，每个线程负责4个元素
// thr_layout：(32,8):(1,32)
// val_layout: (4,1):(1.4)
raked_product( (32,8):(1,32),  (4,1):(1,4) )
    = ((4,32),(1,8)) : ((1,4),(0,128))
  M = (4 个值, 32 个线程) N = (1 个值, 8 个线程)

tile的M被分解成了"4值 x 32线程",N被分解成了"1值 x 8线程"，线程(t_m, t_n)拥有4*t_m + 128*t_n开始的4个连续元素

layout_mn是"站在tile坐标看谁管哪"，但是TiledCopy需要知道"(thr_idx, val_idx) -> (M,N)"，right_inverse负责完成这一映射翻转，(256,4) → (128,8) 内坐标

这个 TiledCopy 一次覆盖的数据 tile 形状 (128,8)
```

## make_cotiled_copy：从数据布局构造
当不关心线程/值到坐标的具体映射，而更关心向量化宽度和偏移时使用
```c++
/** 从线程和值偏移映射构造 TiledCopy。
 * TV 布局将线程和值映射到 data_layout 的余域。
 * 当线程和值不关心拥有特定坐标，而更关心向量化宽度和偏移时有用。
 */
template <class... Args, class AtomTVLayout, class DataLayout>
CUTE_HOST_DEVICE constexpr
auto
make_cotiled_copy(Copy_Atom<Args...> const& copy_atom,
                  AtomTVLayout const& atom_tv_layout,   // atom (thr,val) -> data addr
                  DataLayout   const& data_layout)      // coord -> data addr
{
  static_assert(is_static<AtomTVLayout>::value);
  static_assert(is_static<DataLayout>::value);

  // data addr -> data coord（逆布局，附加 1:0 处理越界）
  auto inv_data_layout = make_layout(left_inverse(data_layout), Layout<_1,_0>{});

  // (tid,vid) -> data_coord
  auto layout_tv_data = composition(inv_data_layout, atom_tv_layout);

  // 验证有效性：AtomTVLayout 指向的内存确实存在于 DataLayout 中
  CUTE_STATIC_ASSERT_V(
      coalesce(composition(make_layout(data_layout, Layout<_1,_0>{}), layout<1>(layout_tv_data)))
      == coalesce(layout<1>(atom_tv_layout)),
      "The memory pointed to by AtomTVLayout does not exist in the DataLayout.");

  // 生成 Tiler 和 Layout_TV ...
  // （省略具体计算，与 make_tiled_copy 类似但基于数据布局）
}
```

## make_tiled_copy_S / make_tiled_copy_D：匹配已有 TiledCopy
```c++
// 构造一个 Src 布局匹配 tiled_copy 的 TiledCopy
template <class... Args, class TiledCopy>
CUTE_HOST_DEVICE
auto
make_tiled_copy_S(Copy_Atom<Args...> const& copy_atom,
                  TiledCopy          const& tiled_copy)
{
  return make_tiled_copy_impl(copy_atom,
                              tiled_copy.get_layoutS_TV(),
                              typename TiledCopy::Tiler_MN{});
}

// 构造一个 Dst 布局匹配 tiled_copy 的 TiledCopy
template <class... Args, class TiledCopy>
CUTE_HOST_DEVICE
auto
make_tiled_copy_D(Copy_Atom<Args...> const& copy_atom,
                  TiledCopy          const& tiled_copy)
{
  return make_tiled_copy_impl(copy_atom,
                              tiled_copy.get_layoutD_TV(),
                              typename TiledCopy::Tiler_MN{});
}

```
当需要用不同的`Copy_Atom`在同一个数据流上操作时（如先`cp.async`加载到smem，再用`ldmatrix`从smem加载到寄存器），这两个函数确保新的`TiledCopy`与已有的`TiledCopy`在 Src 或 Dst 布局上对齐

## make_tiled_copy_A/B/C
让Copy的值序恰好满足`MMA`的喂入顺序，可以直接从`TiledMMA`构造`TiledCopy`
```c++
// 构造匹配 MMA 的 A 矩阵布局的 TiledCopy
template <class... CArgs, class... MArgs>
CUTE_HOST_DEVICE
auto constexpr
make_tiled_copy_A(Copy_Atom<CArgs...> const& copy_atom,
                  TiledMMA<MArgs...>  const& mma)
{
  return make_tiled_copy_impl(copy_atom,
                              mma.get_layoutA_TV(),    // MMA 的 A 布局
                              make_shape(tile_size<0>(mma), tile_size<2>(mma)));  // (M,K)
}

// 构造匹配 MMA 的 B 矩阵布局的 TiledCopy
template <class... CArgs, class... MArgs>
CUTE_HOST_DEVICE
auto constexpr
make_tiled_copy_B(Copy_Atom<CArgs...> const& copy_atom,
                  TiledMMA<MArgs...>  const& mma)
{
  return make_tiled_copy_impl(copy_atom,
                              mma.get_layoutB_TV(),    // MMA 的 B 布局
                              make_shape(tile_size<1>(mma), tile_size<2>(mma)));  // (N,K)
}

// 构造匹配 MMA 的 C 矩阵布局的 TiledCopy
template <class... CArgs, class... MArgs>
CUTE_HOST_DEVICE
auto
make_tiled_copy_C(Copy_Atom<CArgs...> const& copy_atom,
                  TiledMMA<MArgs...>  const& mma)
{
  return make_tiled_copy_impl(copy_atom,
                              mma.get_layoutC_TV(),    // MMA 的 C 布局
                              make_shape(tile_size<0>(mma), tile_size<1>(mma)));  // (M,N)
}

```

## make_tiled_copy_C_atom：原子级 C 拷贝
截断 LayoutC_TV 到一个 atom 的 V
用途：当 epilogue 需要将 C 矩阵的子分块存回时，这个函数构造一个最小的 TiledCopy，其 V 布局与 Copy_Atom 大小匹配，但保持 MMA 的 C 布局顺序
```c++
// 返回能 retile LayoutC_TV 的最小 tiled copy
// 用于流水线 epilogue 的子分块存储
template <class... CArgs, class... MArgs>
CUTE_HOST_DEVICE
auto
make_tiled_copy_C_atom(Copy_Atom<CArgs...> const& copy_atom,
                       TiledMMA<MArgs...>  const& mma)
{
  // 截断 V-layout 到 Copy_Atom 大小，保留 V-order
  auto layoutC_TV = mma.get_layoutC_TV();
  auto copy_V     = Int<Copy_Atom<CArgs...>::NumValSrc>{};
  CUTE_STATIC_ASSERT_V(copy_V <= size<1>(layoutC_TV));
  auto layout_TV  = composition(layoutC_TV, make_layout(make_shape(size<0>(layoutC_TV), copy_V)));

  // 重新计算 Tiler 和 TV 布局 ...
  // （省略具体计算）

  return make_tiled_copy_impl(copy_atom, layout_tv, tiler);
}
```

# 具体例子。
两矩阵相加，用下面3种方案拷贝。
* 直接拷贝到寄存器
* cp.async拷贝到smem
* TMA拷贝

## G2S拷贝
## vecadd_g2r
```c++
#include <cstdio>
#include <vector>
#include <cute/tensor.hpp>

using namespace cute;

template <class TA, class TB, class TC, class TiledCopy>
__global__ void vecadd_g2r_kernel(TA tA, TB tB, TC tC, TiledCopy tiled_copy){
  Tensor tile_A = tA(make_coord(_, _), blockIdx.x, blockIdx.y);
  Tensor tile_B = tB(make_coord(_, _), blockIdx.x, blockIdx.y);
  Tensor tile_C = tC(make_coord(_, _), blockIdx.x, blockIdx.y);

  auto thr_copy = tiled_copy.get_slice(threadIdx.x);
  Tensor thr_A = thr_copy.partition_S(tile_A); // (FrgV, FrgM, RestN)
  Tensor thr_B = thr_copy.partition_S(tile_B);
  Tensor thr_C = thr_copy.partition_D(tile_C);

  Tensor frag_A = make_fragment_like(thr_A);
  Tensor frag_B = make_fragment_like(thr_B);
  Tensor frag_C = make_fragment_like(thr_C);

  copy(tiled_copy, thr_A, frag_A);
  copy(tiled_copy, thr_B, frag_B);

  for(int i = 0; i < size(frag_C); ++i){
    frag_C(i) = frag_A(i) + frag_B(i);
  }

  copy(tiled_copy, frag_C, thr_C);
}

int main(){
  constexpr int M = 256, N = 512;
  auto tensor_shape = make_shape(Int<M>{}, Int<N>{});

  std::vector<float> h_A(M * N), h_B(M * N), h_C(M * N), h_ref(M * N);
  for(int i = 0; i < M * N; i++){
    h_A[i] = float(i % 100);
    h_B[i] = float((2 * i) % 99);
    h_ref[i] = h_A[i] + h_B[i];
  }

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, M * N * sizeof(float));
  cudaMalloc(&d_B, M * N * sizeof(float));
  cudaMalloc(&d_C, M * N * sizeof(float));
  cudaMemcpy(d_A, h_A.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

  Tensor A = make_tensor(make_gmem_ptr(d_A), make_layout(tensor_shape));
  Tensor B = make_tensor(make_gmem_ptr(d_B), make_layout(tensor_shape));
  Tensor C = make_tensor(make_gmem_ptr(d_C), make_layout(tensor_shape));

  // local_tile
  auto tile_shape = make_shape(Int<128>{}, Int<64>{});
  auto tA = tiled_divide(A, tile_shape);  // ((128,64),2,8)
  auto tB = tiled_divide(B, tile_shape);
  auto tC = tiled_divide(C, tile_shape);

  // TV 256个线程排(32,8)，每个线程4个值排(4,1)，因为ABC都是列主序，所以每个线程访问连续的16B元素
  Layout thr_layout = make_layout(make_shape(Int<32>{}, Int<8>{}));
  Layout val_layout = make_layout(make_shape(Int<4>{}, Int<1>{}));

  using CopyOp = UniversalCopy<uint_byte_t<16>>;  // 16B通用拷贝
  using Atom = Copy_Atom<CopyOp, float>;
  TiledCopy tiled_copy = make_tiled_copy(Atom{}, thr_layout, val_layout);

  dim3 grid(size<1>(tA), size<2>(tB));  // (2, 8)
  dim3 block(size(thr_layout));         // (32,8)

  vecadd_g2r_kernel<<<grid, block>>>(tA, tB, tC, tiled_copy);

  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess){
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    return 1;
  }

  cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  bool ok = true;
  for(int i = 0; i < M * N; ++i){
    if(h_ref[i] != h_C[i]){
      ok = false;
      break;
    }
  }

  printf(ok ? "PASS: C = A + B (%d x %d)\n" : "FAIL!\n", M, N);
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
  return ok ? 0 : 1;
}
```


## cp_async
```c++
#include <cstdio>
#include <vector>
#include <cute/tensor.hpp>
using namespace cute;

template <class SmemLayout>
struct SharedStorage{
  cute::ArrayEngine<float, cute::cosize_v<SmemLayout>> a;
  cute::ArrayEngine<float, cute::cosize_v<SmemLayout>> b;
};


template<class TA, class TB, class TC, class TiledCopyG2S, class TiledCopyS2R, class SmemLayout>
__global__ void vecadd_cp_async_kernel(TA tA, TB tB, TC tC, 
                                      TiledCopyG2S copy_g2s,
                                      TiledCopyS2R copy_s2r,
                                      SmemLayout smem_layout){

  extern __shared__ char shared_memory[];
  SharedStorage<SmemLayout>& shared = *reinterpret_cast<SharedStorage<SmemLayout>*>(shared_memory);

  Tensor sA = make_tensor(make_smem_ptr(shared.a.begin()), smem_layout);
  Tensor sB = make_tensor(make_smem_ptr(shared.b.begin()), smem_layout);

  // local_tile
  Tensor tile_A = tA(make_coord(_, _), blockIdx.x, blockIdx.y);
  Tensor tile_B = tB(make_coord(_, _), blockIdx.x, blockIdx.y);
  Tensor tile_C = tC(make_coord(_, _), blockIdx.x, blockIdx.y);

  // 1、cp.async gemm to smem
  auto thr_g2s = copy_g2s.get_thread_slice(threadIdx.x);
  Tensor tAgA = thr_g2s.partition_S(tile_A); // ((_4,_1),_1,_8)
  Tensor tAsA = thr_g2s.partition_S(sA);
  Tensor tBgB = thr_g2s.partition_S(tile_B);
  Tensor tBsB = thr_g2s.partition_S(sB);

  // g2s copy
  copy(copy_g2s, tAgA, tAsA);
  copy(copy_g2s, tBgB, tBsB);

  // commit & wait
  cp_async_fence();   // commit_group
  cp_async_wait<0>();
  __syncthreads();

  // smem to frag
  auto thr_s2r = copy_s2r.get_thread_slice(threadIdx.x);
  Tensor tsA = thr_s2r.partition_S(sA);
  Tensor tsB = thr_s2r.partition_S(sB);
  Tensor tsC = thr_s2r.partition_D(tile_C);

  Tensor frA = make_fragment_like(tsA);
  Tensor frB = make_fragment_like(tsB);
  Tensor frC = make_fragment_like(tsA);

  copy(copy_s2r, tsA, frA);
  copy(copy_s2r, tsB, frB);

  for(int i = 0; i < size(frC); ++i){
    frC(i) = frA(i) + frB(i);
  }

  copy(copy_s2r, frC, tsC);
}

int main(){

  constexpr int M = 256, N = 512;
  auto tensor_shape = make_shape(Int<M>{}, Int<N>{});

  std::vector<float> h_A(M * N), h_B(M * N), h_C(M * N), h_ref(M * N);
  for (int i = 0; i < M * N; ++i) {
    h_A[i] = float(i % 100);
    h_B[i] = float((2 * i) % 97);
    h_ref[i] = h_A[i] + h_B[i];
  }

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, M * N * 4);
  cudaMalloc(&d_B, M * N * 4);
  cudaMalloc(&d_C, M * N * 4);
  cudaMemcpy(d_A, h_A.data(), M * N * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B.data(), M * N * 4, cudaMemcpyHostToDevice);

  Tensor A = make_tensor(make_gmem_ptr(d_A), make_layout(tensor_shape)); // (256, 512) : (1, 256)
  Tensor B = make_tensor(make_gmem_ptr(d_B), make_layout(tensor_shape));
  Tensor C = make_tensor(make_gmem_ptr(d_C), make_layout(tensor_shape));

  auto block_shape = make_shape(Int<128>{}, Int<64>{});
  auto tA = tiled_divide(A, block_shape);  // ((128,64), 2, 8)
  auto tB = tiled_divide(B, block_shape);
  auto tC = tiled_divide(C, block_shape);

  Layout thr_layout = make_layout(make_shape(Int<32>{}, Int<8>{}));
  Layout val_layout = make_layout(make_shape(Int<4>{},  Int<1>{}));

  using CopyOpG2S = SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>;
  TiledCopy copy_g2s = make_tiled_copy(Copy_Atom<CopyOpG2S, float>{}, thr_layout, val_layout);

  // smem→reg→gmem 沿用路径 A 的 UniversalCopy 16B 方案
  using CopyOpS2R = UniversalCopy<uint_byte_t<16>>;
  TiledCopy copy_s2r = make_tiled_copy(Copy_Atom<CopyOpS2R, float>{}, thr_layout, val_layout);

  using SmemLayout = decltype(make_layout(Shape<_128, _64>{}));  // (128,64):(1,128)，与 gmem tile 同构
  auto smem_layout = make_layout(Shape<_128, _64>{});

  dim3 grid(size<1>(tA), size<2>(tA));
  dim3 blk(size(thr_layout));
  int smem_size = int(sizeof(SharedStorage<SmemLayout>));
  printf("  动态 smem = %d 字节（>48KB，需 opt-in）\n", smem_size);

  auto kfn = &vecadd_cp_async_kernel<decltype(tA), decltype(tB), decltype(tC),
                                    decltype(copy_g2s), decltype(copy_s2r), SmemLayout>;
  cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  kfn<<<grid, blk, smem_size>>>(tA, tB, tC, copy_g2s, copy_s2r, smem_layout);
  cudaDeviceSynchronize();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

  cudaMemcpy(h_C.data(), d_C, M * N * 4, cudaMemcpyDeviceToHost);
  bool ok = true;
  for (int i = 0; i < M * N; ++i) if (h_C[i] != h_ref[i]) { ok = false; break; }
  printf(ok ? "PASS: C = A + B (%d x %d)\n" : "FAIL!\n", M, N);
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
  return ok ? 0 : 1;
}
```

## TMA
```c++
#include <cstdio>
#include <vector>
#include <cute/tensor.hpp>
using namespace cute;

template <class SmemLayout>
struct SharedStorage{
    cute::ArrayEngine<float, cute::cosize_v<SmemLayout>> a;
    cute::ArrayEngine<float, cute::cosize_v<SmemLayout>> b;
    cute::ArrayEngine<float, cute::cosize_v<SmemLayout>> c;
    alignas(16) cute::uint64_t tma_mbar[1];     // mbarrier 占用smme的8字节
};

template <class TmaCopyA, class TmaCopyB,
            class TmaCopyC, class TiledCopyUni,
            class SmemLayout, class GShape, class CTA_Tiler>
__global__ void vecadd_tma_kernel(CUTE_GRID_CONSTANT TmaCopyA const tma_a,
                                  CUTE_GRID_CONSTANT TmaCopyB const tma_b,
                                  CUTE_GRID_CONSTANT TmaCopyC const tma_c,
                                  TiledCopyUni copy_uni, SmemLayout smem_layout,
                                  GShape g_shape, CTA_Tiler cta_tiler){
    extern __shared__ char shared_memory[];
    SharedStorage<SmemLayout>& shared = *reinterpret_cast<SharedStorage<SmemLayout>*>(shared_memory);
    Tensor sA = make_tensor(make_smem_ptr(shared.a.begin()), smem_layout);
    Tensor sB = make_tensor(make_smem_ptr(shared.b.begin()), smem_layout);
    Tensor sC = make_tensor(make_smem_ptr(shared.c.begin()), smem_layout);
    uint64_t* tma_mbar = shared.tma_mbar;

    Tensor mA = tma_a.get_tma_tensor(g_shape);
    Tensor mB = tma_b.get_tma_tensor(g_shape);
    Tensor mC = tma_c.get_tma_tensor(g_shape);

    Tensor gA = flat_divide(mA, cta_tiler); // ((128,64), 2, 8)
    Tensor gB = flat_divide(mB, cta_tiler);
    Tensor gC = flat_divide(mC, cta_tiler);

    // 本 CTA 的 tile 在 (2,8) 网格中的线性位置
    int tile_idx = blockIdx.x + blockIdx.y * 2;

    // 1、tma load，单线程发起两条TMA指令，A、B共用一个mbarrier
    auto cta_tma_a = tma_a.get_slice(Int<0>{}); // 单CTA，无cluster
    Tensor tAgA_x = cta_tma_a.partition_S(gA);  // (TMA, RestM, RestN)
    Tensor tAsA_x = cta_tma_a.partition_D(sA);  // (TMA)
    Tensor tBgB_x = tma_b.get_slice(Int<0>{}).partition_S(gB);
    Tensor tBsB_x = tma_b.get_slice(Int<0>{}).partition_D(sB);

    Tensor tAgA = group_modes<1,rank(tAgA_x)>(tAgA_x);   // (TMA, REST)
    Tensor tAsA = group_modes<1,rank(tAsA_x)>(tAsA_x);   // (TMA, 1)
    Tensor tBgB = group_modes<1,rank(tBgB_x)>(tBgB_x);
    Tensor tBsB = group_modes<1,rank(tBsB_x)>(tBsB_x);
    static_assert(size<1>(tAsA) == 1);  // smem 侧一个 stage 装满一个 tile

    // store 侧：S = smem，D = gmem（方向与 load 相反）
    auto cta_tma_c = tma_c.get_slice(Int<0>{});
    Tensor tCsC_x = cta_tma_c.partition_S(sC);           // (TMA)
    Tensor tCgC_x = cta_tma_c.partition_D(gC);           // (TMA, RestM, RestN)
    Tensor tCsC = group_modes<1,rank(tCsC_x)>(tCsC_x);   // (TMA, 1)
    Tensor tCgC = group_modes<1,rank(tCgC_x)>(tCgC_x);   // (TMA, REST)
    static_assert(size<1>(tCsC) == 1);

    constexpr int kTmaTransactionBytes = sizeof(make_tensor_like(tensor<0>(tAsA)));  // 32KB

    if(threadIdx.x == 0){
        tma_mbar[0] = 0;
        initialize_barrier(tma_mbar[0], 1 /*期望到达线程数*/);
        // 两条 TMA 的字节一起挂上 barrier：arrive + expect_tx(64KB)
        set_barrier_transaction_bytes(tma_mbar[0], 2 * kTmaTransactionBytes);
        copy(tma_a.with(tma_mbar[0]), tAgA(_, tile_idx), tAsA(_, 0));
        copy(tma_b.with(tma_mbar[0]), tBgB(_, tile_idx), tBsB(_, 0));
    }
    __syncthreads();
    wait_barrier(tma_mbar[0], 0 /*parse*/); // 当拷贝完成的时候，会翻转到1，这里的语义是所有线程等待直到相位不为0

    // 2、SMEM to Reg
    auto thr_copy = copy_uni.get_thread_slice(threadIdx.x);
    Tensor tsA = thr_copy.partition_S(sA);
    Tensor tsB = thr_copy.partition_S(sB);
    Tensor tsC = thr_copy.partition_D(sC);

    Tensor frA = make_fragment_like(tsA);
    Tensor frB = make_fragment_like(tsA);
    Tensor frC = make_fragment_like(tsA);

    copy(copy_uni, tsA, frA);
    copy(copy_uni, tsB, frB);

    for (int i = 0; i < size(frC); ++i) {
        frC(i) = frA(i) + frB(i);
    }
    copy(copy_uni, frC, tsC);

    //
    // 3. TMA STORE：generic 代理的 smem 写 → fence → 单线程一条 TMA 发走
    //
    tma_store_fence();          // fence.proxy.async.shared::cta
    __syncthreads();
    if (threadIdx.x == 0) {
        copy(tma_c, tCsC(_, 0), tCgC(_, tile_idx));   // 需要 tCsC/tCgC，见下方定义
    }
    tma_store_wait<0>();
    __syncthreads();
}

int main(){
    constexpr int M = 256, N = 512;
    auto tensor_shape = make_shape(Int<M>{}, Int<N>{});

    std::vector<float> h_A(M * N), h_B(M * N), h_C(M * N), h_ref(M * N);
    for (int i = 0; i < M * N; ++i) {
        h_A[i] = float(i % 100);
        h_B[i] = float((2 * i) % 97);
        h_ref[i] = h_A[i] + h_B[i];
    }


    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * N * 4);
    cudaMalloc(&d_B, M * N * 4);
    cudaMalloc(&d_C, M * N * 4);
    cudaMemcpy(d_A, h_A.data(), M * N * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), M * N * 4, cudaMemcpyHostToDevice);

    Tensor A = make_tensor(make_gmem_ptr(d_A), make_layout(tensor_shape));
    Tensor B = make_tensor(make_gmem_ptr(d_B), make_layout(tensor_shape));
    Tensor C = make_tensor(make_gmem_ptr(d_C), make_layout(tensor_shape));

    auto block_shape = make_shape(Int<128>{}, Int<64>{});
    auto smem_layout = make_layout(Shape<_128, _64>{});   // (128,64):(1,128)


    // host 侧生成 TensorMap：load 两个、store 一个
    auto tma_a = make_tma_copy(SM90_TMA_LOAD{}, A, smem_layout, block_shape, Int<1>{});
    auto tma_b = make_tma_copy(SM90_TMA_LOAD{}, B, smem_layout, block_shape, Int<1>{});
    auto tma_c = make_tma_copy(SM90_TMA_STORE{}, C, smem_layout, block_shape, Int<1>{});

    Layout thr_layout = make_layout(make_shape(Int<32>{}, Int<8>{}));
    Layout val_layout = make_layout(make_shape(Int<4>{},  Int<1>{}));
    using CopyOpUni = UniversalCopy<uint_byte_t<16>>;
    TiledCopy copy_uni = make_tiled_copy(Copy_Atom<CopyOpUni, float>{}, thr_layout, val_layout);


    dim3 grid(2, 8);
    dim3 blk(size(thr_layout));
    int smem_size = int(sizeof(SharedStorage<decltype(smem_layout)>));
    printf("  动态 smem = %d 字节（3 x 32KB tile + mbarrier）\n", smem_size);

    auto kfn = &vecadd_tma_kernel<decltype(tma_a), decltype(tma_b), decltype(tma_c),
                                    decltype(copy_uni), decltype(smem_layout),
                                    decltype(tensor_shape), decltype(block_shape)>;
    cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    kfn<<<grid, blk, smem_size>>>(tma_a, tma_b, tma_c, copy_uni, smem_layout, tensor_shape, block_shape);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

    cudaMemcpy(h_C.data(), d_C, M * N * 4, cudaMemcpyDeviceToHost);
    bool ok = true;
    for (int i = 0; i < M * N; ++i) if (h_C[i] != h_ref[i]) { ok = false; break; }
    printf(ok ? "PASS: C = A + B (%d x %d)\n" : "FAIL!\n", M, N);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return ok ? 0 : 1;
}
```