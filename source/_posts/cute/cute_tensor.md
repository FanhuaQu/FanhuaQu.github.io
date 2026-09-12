---
title: Cute Tensor
date: 2026-08-18 20:00:00
tags: [cute, Layout, Tensor]
categories: [Cute 学习笔记]
description: 文章介绍了 cute Tensor
---

# Cute Tensor
Tensor是Cute的核心容器，抽象了数组元素的组织方式和数组元素的存储细节。可以简单理解成 `Tensor = 数据 + Layout`

Tensor由两个模板参数组成，Engine & Layout，前者用于管理和访问tensor中的数据，类似于std::vector里面的迭代器，Layout用于描述tensor中元素的布局。

Tensor支持kernel开发中的任意内存层级，包括host memory、global memory、shared memory和寄存器内存等。

# 基础操作
cute Tensor的定义在`include\cute\tensor_impl.hpp`

Tensor提供了类似于容器的操作运算：
* .data() 返回Tensor的首元素地址
* .size() 返回Tensor的大小，return cute::size(shape())
* .operator[] 通过坐标访问Tensor中的元素
* .operator()(Coord const& coord) 也是通过坐标访问Tensor中的元素
* .operator()(Coords…) 支持多维坐标访问

同时提供了Layout支持的操作。等价于处理tensor.layout()
* rank
* depth
* shape
* size
* layout
* tensor

# Tensor engine
可以简单理解为迭代器
```c++
template <int S, class T, size_t N>
struct ArrayEngine<sparse_elem<S,T>, N>
{
  static_assert(N % S == 0, "Expected a multiple of the sparsity.");
  using value_type   = sparse_elem<S,T>;
  using Storage      = typename conditional<(sizeof_bits<T>::value % 8 == 0),
                                            array_aligned<T,N/S>,
                                            array_subbyte<T,N/S>>::type;
  using iterator     = sparse_ptr<S,sparse_elem<S,T>*>;
  using reference    = typename iterator_traits<iterator>::reference;
  using element_type = typename iterator_traits<iterator>::element_type;
  Storage storage_;

  CUTE_HOST_DEVICE constexpr auto begin() const { return recast_ptr<value_type>(storage_.begin()); }
  CUTE_HOST_DEVICE constexpr auto begin()       { return recast_ptr<value_type>(storage_.begin()); }
};
```

# 创建 Tensor
cute 中的Tensor分成两类
* 栈上Tensor(owning tensor)：负责数据的生命周期(内存分配和释放)，拷贝时进行深拷贝，生成独立的全新副本
* 堆上Tensor(nonowning tensor)：仅仅作为现有数据的一种视图。复制时仅传递同一内存的引用，析构时不释放底层内存。

在 cute 中通过make_tensor()来创建tensor，
```c++
// Make an owning Tensor that will allocate a static array
// e.g. make_tensor<float>(Int<12>{})
template <class T, class... Args>
CUTE_HOST_DEVICE constexpr
auto
make_tensor(Args const&... args)
{
  static_assert((not has_dereference<Args>::value && ...), "Expected layout args... in make_tensor<T>(args...)");
  return MakeTensor<T>{}(args...);
}

// Make a non-owning Tensor that will use a pointer (view)
// e.g. make_tensor(vec.data(), 12)
template <class Iterator, class... Args>
CUTE_HOST_DEVICE constexpr
auto
make_tensor(Iterator const& iter, Args const&... args)
{
  static_assert(has_dereference<Iterator>::value, "Expected iterator iter in make_tensor(iter, args...)");
  static_assert((not has_dereference<Args>::value && ...), "Expected layout args... in make_tensor(iter, args...)");
  return MakeTensor<Iterator>{}(iter, args...);
}
```

# Nonowning Tensor
用户自己申请显存，结合特定的Layout创建的Tensor。

传入的参数是一个指针和一个Layout
特别的，可以通过`make_gmem_ptr(g)`和`make_smem_ptr(s)`来指明指针指向的数组是在全局内存还是在共享内存

下面是一些用例
```c++
float* A = ...;

// make_tensor里面会调用MakeTensor()，支持传入不同的参数
Tensor tensor_8 = make_tensor(A, make_layout(Int<8>{}));    // 传入layout构造
Tensor tensor_8_s = make_tensor(A, Int<8>{});   // 传入shape构造
Tensor tensor_8_ss = make_tensor(A, Int<8>{}, Int<2>{}); // shape & stride，实际上里面会构造layout


// 指定Global memory
Tensor gmem_8s = make_tensor(make_gmem_ptr(A), Int<8>{});
Tensor gmem_8d = make_tensor(make_gmem_ptr(A), 8);

// shared memory
__shared__ float smem[8];
Tensor smem_8s = make_tensor(make_smem_ptr(smem), Int<8>{});
Tensor smem_8d = make_tensor(make_smem_ptr(smem), 8);
```

通过`make_smem_ptr`和`make_gmem_ptr`可以指明指针指向的内存空间，Layout则是用于解释内存布局

# Owning tensor
这种cute是用于自己创建数据buffer和layout来生成tensor，通常在寄存器层面创建(很好理解，寄存器tensor负责自己的生命周期，用完就应该释放寄存器空间)

下面是一些例子：
```c++
Tensor rmem_4x8_col = make_tensor<float>(Shape<_4, _8>{});
Tensor rmem_4x8_row = make_tensor<float>(Shape<_4, _8>{}, LayoutRight{});
Tensor rmem_4x8_pad = make_tensor<float>(Shape<_4, _8>{}, Stride<_32, _2>{});
Tensor rmem_4x8_like = make_tensor_like(rmem_4x8_pad);
```
上面的pad版本，`cosize(layout) = 111`, 也就是实际占用了111个寄存器空间，但是这个Tensor视角只看到其中的4 x 8个元素。

`make_tensor_like`函数创建的tensor的数据类型与形状和输入的tensor和输入的tensor相同，并尝试使用相同的stride。(就是按照原tensor的形状另起一块**紧凑布局**)  
使用cute::print()打印上面4个tensor如下
```bash
ptr[32b](0x7f5c41fff990) o (_4,_8):(_1,_4)
ptr[32b](0x7f5c41fffa10) o (_4,_8):(_8,_1)
ptr[32b](0x7f5c41fffa90) o (_4,_8):(_32,_2)
ptr[32b](0x7f5c41fffc50) o (_4,_8):(_8,_1)
```

## make_fragment_like
创建一个形状保持一样，但是把mode-0强制设置成列主序(stride=1)紧凑布局
```c++
  template <class Shape, class Stride>
  auto make_fragment_like(Layout<Shape,Stride> const& layout)
  {
    constexpr int R = Layout<Shape,Stride>::rank;             // 秩
    if constexpr (R > 1 && is_static<Shape>::value) {
      return tiled_product(
          // [1] mode-0：强制 LayoutLeft、stride-1 的紧凑块
          make_layout(get<0>(layout.shape()),
                      compact_major<LayoutLeft>(filter_zeros(get<0>(layout.stride()),
  get<0>(layout.shape())))),
          // [2] 其余模式：按原 stride 的大小顺序重排成紧凑布局
          make_ordered_layout(take<1,R>(layout.shape()), take<1,R>(layout.stride())));
    } else {
      return make_layout(layout.shape());                      // 秩<=1 或动态shape：退化为普通紧凑
    }
  }
```
看例子：
```c++
Tensor rmem_4x8_pad = make_tensor<float>(Shape<_4, _8>{}, Stride<_32, _2>{});
Tensor rmem_4x8_freg_like = make_fragment_like(rmem_4x8_pad);
```
打印的结果如下：
```bash
ptr[32b](0x7fa6adfffa10) o (_4,_8):(_32,_2)
ptr[32b](0x7fa6adfffc50) o (_4,(_8)):(_1,(_4))
```
在cutlass里面，一个MMA或者copy指令的的寄存器tile就叫做fragment，MMA/Copy的硬件布局通常是需要mode-0是连续并且列主序的，后面整理tiled_copy的时候再看

## make_coord_tensor
根据输入的layout创建一个tensor，元素是layout的offset
```c++
Tensor rmem_4x8_cood = make_coord_tensor(make_layout(make_shape(_4{}, _6{}), make_stride(_2{}, _8{})));
ArithTuple_0 o (_4,_6):(_2,_8):
    0    8   16   24   32   40
    2   10   18   26   34   42
    4   12   20   28   36   44
    6   14   22   30   38   46
```

## make_identity_tensor
创建一个元素是自身坐标的Tensor，通常用于边界位置判断和TMA数据加载
```c++
Tensor tensor_identity = make_identity_tensor(Shape<_4, _8>{});

ArithTuple(_0,_0) o (_4,_8):(_1@0,_1@1):
  (0,0)  (0,1)  (0,2)  (0,3)  (0,4)  (0,5)  (0,6)  (0,7)
  (1,0)  (1,1)  (1,2)  (1,3)  (1,4)  (1,5)  (1,6)  (1,7)
  (2,0)  (2,1)  (2,2)  (2,3)  (2,4)  (2,5)  (2,6)  (2,7)
  (3,0)  (3,1)  (3,2)  (3,3)  (3,4)  (3,5)  (3,6)  (3,7)
```

# 访问 Tensor
可以通过`operator()`和`operator[]`访问tensor的元素，接受元组类型的逻辑坐标。`operator()`还支持使用`_`进行切片(`slice`)访问
```c++
  template <class Coord>
  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  operator[](Coord const& coord) {
    return data()[layout()(coord)];
  }
```
下面是一些例子：
```c++
Tensor tensor_a = make_coord_tensor(make_layout(Shape<_4, _6>{}, Stride<_2, _8>{}));
ArithTuple_0 o (_4,_6):(_2,_8):
    0    8   16   24   32   40
    2   10   18   26   34   42
    4   12   20   28   36   44
    6   14   22   30   38   46

int val_6 = tensor_a[6];    // 12
int val_2_3 = tensor_a(2,3);    // 28

auto tensor_a_col1 = tensor_a(_,0);
/*
    ArithTuple0 o (_4):(_2):
        0
        2
        4
        6
*/
```

# Tensor Algebra
Layout Algebra中的操作同样可以作用于Tensor，本质上就是对Tensor的Layout进行处理。
```c++
composition(Tensor, Tiler)
logical_divide(Tensor, Tiler)
zipped_divide(Tensor, Tiler)
tiled_divide(Tensor, Tiler)
flat_divide(Tensor, Tiler)
```


# Partitioning a Tensor
通过composition、tiling和slice等操作实现对 tensor 的分块，包括内部分区`(inner-partitioning)`，外部分区`(outer-partitioning)`和thread-value分区`(TV-layout-partitioning)`

先来回顾一下zipped_divide的“内外二分”
所有的partition都来自于zipped_divide，其思想如下：
> 把一个tensor的modes沿着tiler切成两部分，块内的"Tile 模式"和块之间的"Rest 模式"
原始tensor(M, N), 对tile(m, n)进行zipped_divide操作
结果的形式是(tile内(m,n), tile间(M/m, N/n))
举个具体的例子：
```c++
Layout L1 = make_layout(Shape<_4,_6>{});

/*
(_4,_6):(_1,_4)
       0    1    2    3    4    5 
    +----+----+----+----+----+----+
 0  |  0 |  4 |  8 | 12 | 16 | 20 |
    +----+----+----+----+----+----+
 1  |  1 |  5 |  9 | 13 | 17 | 21 |
    +----+----+----+----+----+----+
 2  |  2 |  6 | 10 | 14 | 18 | 22 |
    +----+----+----+----+----+----+
 3  |  3 |  7 | 11 | 15 | 19 | 23 |
    +----+----+----+----+----+----+
*/

Layout tiler = make_layout(Shape<_2,_2>{}, Stride<_1,_4>{});
/*
(_2,_2):(_1,_4)
      0   1 
    +---+---+
 0  | 0 | 4 |
    +---+---+
 1  | 1 | 5 |
    +---+---+
*/

// ((_2,_2),(_2,_3)):((_1,_4),(_2,_8))
auto res = zipped_divide(L1, tiler);
```
上面结果是rank-2的layout，mode-0是块内部，mode-1是块摆放的网格。
两种取法 inner / outer
## inner-partitioning
在整个mode-0(也就是tile上)取`_`(也就是全要)，在mode-1(Rest)上用cood切→ “其实就是取出大Tensor上的某个tile”
```c++
Layout L1 = make_layout(Shape<_4,_6>{});
auto T1 = make_coord_tensor(L1);
auto tile_1_0 = inner_partition(T1, tiler, make_coord(_1{},_0{}));

/*
ArithTuple_0 o (_4,_6):(_1,_4):
    0    4    8   12   16   20
    1    5    9   13   17   21
    2    6   10   14   18   22
    3    7   11   15   19   23

ArithTuple_2 o (_2,_2):(_1,_4):
    2    6
    3    7
*/
```

## outer_partition
在mode-1(Rest)上取`_`，在mode-0(tile)上用coord切 → “其实就是拿到所有tile同样index上的数据”
```c++
Layout L1 = make_layout(Shape<_4,_6>{});
auto T1 = make_coord_tensor(L1);
auto index_1_0 = outer_partition(T1, tiler, make_coord(_1{}, _0{}));

/*
ArithTuple_0 o (_4,_6):(_1,_4):
    0    4    8   12   16   20
    1    5    9   13   17   21
    2    6   10   14   18   22
    3    7   11   15   19   23

ArithTuple_1 o (_2,_3):(_2,_8):
    1    9   17
    3   11   19
*/
```
> 总结：inner = 一块 tile 的全部内容；outer = 某个固定索引在一堆 tile 里的分布。

## local_tile CTA级别切分Tile

```c++
template <class Tensor, class Tiler, class Coord,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr
auto
local_tile(Tensor    && tensor,
           Tiler const& tiler,   // tiler to apply
           Coord const& coord)   // coord to slice into "remainder"
{
  return inner_partition(static_cast<Tensor&&>(tensor),
                         tiler,
                         coord);
}

// 4参数重载版本
local_tile(Tensor    && tensor,
           Tiler const& tiler,   // tiler to apply
           Coord const& coord,   // coord to slice into "remainder"
           Proj  const& proj)

```
就是把一个大张量切成 CTA 的 tile，并挑出 (blockIdx.x, blockIdx.y) 那块
4参数版本用一个 Step<...> 把 tiler 里不关心的维度 X 掉。这是处理 A/B/C三个矩阵共用一个 cta_tiler 的惯用法

举例：
```c++
// 基础版本就是和上面一样
Layout L2 = make_layout(Shape<_4,_6>{});
auto T2 = make_coord_tensor(L2);
Layout tiler2 = make_layout(Shape<_2,_2>{}, Stride<_1,_4>{});
auto tile_11 = local_tile(T2, tiler2, make_coord(_1{}, _1{}));
/*
ArithTuple_0 o (_4,_6):(_1,_4):
    0    4    8   12   16   20
    1    5    9   13   17   21
    2    6   10   14   18   22
    3    7   11   15   19   23

ArithTuple_10 o (_2,_2):(_1,_4):
   10   14
   11   15
*/

// 4参数版本
auto A = make_coord_tensor(make_layout(Shape<_4, _6>{}, LayoutRight{}));
auto B = make_coord_tensor(make_layout(Shape<_6, _4>{}, LayoutRight{}));
auto C = make_coord_tensor(make_layout(Shape<_4, _4>{}, LayoutRight{}));
auto cta_tiler = Shape<_2, _3, _2>{};
auto cta_coord = make_coord(_1{}, _1{}, _);

Tensor A_1 = local_tile(A, cta_tiler, cta_coord, Step<_1, X, _1>{});
Tensor B_1 = local_tile(B, cta_tiler, cta_coord, Step<X, _1, _1>{});
Tensor C_1 = local_tile(B, cta_tiler, cta_coord, Step<_1, _1, X>{});

print_tensor(A);printf("\n");
print_tensor(B);printf("\n");
print_tensor(C);printf("\n");
print_tensor(A_1);printf("\n");
print_tensor(B_1);printf("\n");
print_tensor(C_1);printf("\n");

/*
A
ArithTuple_0 o (_4,_6):(_6,_1):
    0    1    2    3    4    5
    6    7    8    9   10   11
   12   13   14   15   16   17
   18   19   20   21   22   23

B
ArithTuple_0 o (_6,_4):(_4,_1):
    0    1    2    3
    4    5    6    7
    8    9   10   11
   12   13   14   15
   16   17   18   19
   20   21   22   23

C
ArithTuple_0 o (_4,_4):(_4,_1):
    0    1    2    3
    4    5    6    7
    8    9   10   11
   12   13   14   15

A分片
ArithTuple_12 o (_2,_2,_3):(_6,_1,_2):
   12   13
   18   19
----------
   14   15
   20   21
----------
   16   17
   22   23

B分片
ArithTuple_12 o (_3,_2,_2):(_4,_1,_2):
   12   13
   16   17
   20   21
----------
   14   15
   18   19
   22   23

C分片
ArithTuple_11 o (_2,_3):(_4,_1):
   11   12   13
   15   16   17
*/
```
上面的结果很好理解。

## local_partition —— 线程级分发
CTA在拿到自己的tile之后，要分配给warp/thread。利用local_partition可以把tile的每个位置映射到线程编号。

```c++
// Tile a tensor according to the flat shape of a layout that provides the coordinate of the target index.
// This is typical at the Thread level where data is partitioned across repeated patterns of threads:
//   Tensor data = ...                                                            // (_16,_64)
//   Tensor thr_data = local_partition(data, Layout<Shape<_2,_16>>{}, thr_idx);   // ( _8, _4)
template <class Tensor, class LShape, class LStride, class Index,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE
auto
local_partition(Tensor                     && tensor,
                Layout<LShape,LStride> const& tile,    // coord -> index
                Index                  const& index)   // index to slice for
{
  static_assert(is_integral<Index>::value);
  return outer_partition(static_cast<Tensor&&>(tensor),
                         product_each(shape(tile)),
                         tile.get_flat_coord(index));
}

// Same as above, but with a projection parameter to strip out unwanted tiling modes for convenience
//   when using projections of the same tiler.
// This is typical at the Thread level where data is partitioned across projected layouts of threads:
//   Tensor dataA = ...                                                            // (M,K)
//   Tensor dataB = ...                                                            // (N,K)
//   Tensor dataC = ...                                                            // (M,N)
//   auto thr_layout = Layout<Shape<_2,_16,_1>, Stride<_16,_1,_0>>{};
//   Tensor thrA = local_partition(dataA, thr_layout, thr_idx, Step<_1, X,_1>{});  // (M/2,K/1)
//   Tensor thrB = local_partition(dataB, thr_layout, thr_idx, Step< X,_1,_1>{});  // (N/16,K/1)
//   Tensor thrC = local_partition(dataC, thr_layout, thr_idx, Step<_1,_1, X>{});  // (M/2,N/16)
template <class Tensor, class LShape, class LStride, class Index, class Projection,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE
auto
local_partition(Tensor                     && tensor,
                Layout<LShape,LStride> const& tile,   // coord -> index
                Index                  const& index,  // index to slice for
                Projection             const& proj)
{
  return local_partition(static_cast<Tensor&&>(tensor),
                         dice(proj, tile),
                         index);
}
```
下面是使用举例：
```c++
// 1、基础版本
// 假如每个CTA每次处理一个tile，线程排布式是2行3列总的6个线程
Tensor tile = make_coord_tensor(make_layout(Shape<_4, _6>{}, LayoutRight{}));
Layout thread_layout = make_layout(Shape<_2, _3>{}, LayoutRight{});

Tensor thr_data0 = local_partition(tile, thread_layout, _0{});
Tensor thr_data3 = local_partition(tile, thread_layout, _3{});

/*
打印结果如下
tile:
ArithTuple_0 o (_4,_6):(_6,_1):
    0    1    2    3    4    5
    6    7    8    9   10   11
   12   13   14   15   16   17
   18   19   20   21   22   23

thr_data0:
ArithTuple_0 o (_2,_2):(_12,_3):
    0    3
   12   15

thr_data1:
ArithTuple_6 o (_2,_2):(_12,_3):
    6    9
   18   21
*/

```
上面的逻辑就是，一个tile按照thread的布局进行zipped_divide，然后可以通过index获取线程每轮处理的元素。

多参数版本也是一样的，比如每个mma每次处理一个16*8的输出，也就是m16n8k16的指令，用local_partition就可以获得每个线程要负责的那小块A、B和输出C。
```c++
auto gmem_ten = local_tile(gmem, tiler, make_coord(bx, by));
auto sA = make_tensor(smemA.data, ...);
auto tA = local_partition(sA, copy_layout_A, thread_idx); // 获取每个线程的smem片段
auto cAcc = local_partition(acc, mma_tiled_MN, thread_idx); // 每个线程的累加器

```

# T-V Layout
在使用cute的时候，我们经常会有这样的场景，一个线程需要处理多个元素，这样就出现了：线程之间有layout，线程内部元素之间也有一个layout。在这种情况下，可以构造一个layout，其mode-0是thread之间的layout，mode-1是一个线程需要处理元素的layout，我们称这种layout为tv-layout。
layout是用于描述逻辑坐标与位置的映射关系，tv-layout则是用于描述坐标(tid, vid)与位置的映射关系。

具体介绍，以SM80_16x8_Row为例，这是Ampere MMA的A矩阵的TV Layout:
```c++
using SM80_16x8_Row = Layout<Shape<Shape<_4,_8>, Shape<_2,_2>>, Stride<Stride<_32,_1>, Stride<_16,_8>>>;
```
* T部分是(_4,_8):(_16,_8), 32个线程，tid按照4行8列行主序排布，这里为什么stride是32呢？
* V部分是(_2,_2):(_16,_8), 这是每个线程持有的4个值

tv-layout是TiledMMA和TiledCopy的基石，一个TiledMMA = “铺了很多份相同的atom”，TV layout就是单份atom的分布，tiled_mma负责将这个atom复制到整个tile网格
1. Alayout(TV): atom内部——给定线性tid和，和线程负责元素的index，就可以知道元素落在MNK的具体位置
2. make_tiled_mma的复制，把这个patten在整个tile上重复，得到tiled_mma.get_thread_slice(yhr_idx)，然后再local_partition出某个线程的片段。
