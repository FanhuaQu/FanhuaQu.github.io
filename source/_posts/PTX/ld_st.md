---
title: ld & st 指令
date: 2026-07-25 20:00:00
tags: [CUDA, PTX, ld, st]
categories: [PTX 学习笔记]
description: 文章介绍了 PTX 中的 缓存运算符， ld 指令和 st 指令。
---

# Cache Operator

在加载或存储指令中缓存运算符仅作为性能提示，在`ld`或`st`指令中，使用缓存运算符不改变程序的内存一致性行为。
下面是一些缓存运算符的语义：
ld
* .ca: 这是默认的缓存操作，会在所有级别(L1 & L2)分配缓存行，并采用正常的缓存驱逐策略。由于L2缓存是所有SM共享的，因此全局一致，L1缓存是每个SM独享的，硬件不会自动同步不同SM的L1中的数据，因此可能存下下面情况
  * SM0的线程先写了某个全局变量(经过L1和L2) ——> SM1的线程紧接着也写了这个全局变量 ———> SM0 的线程读这个变量，就会读到L1缓存的旧值
  * 为了避免这种问题，GPU驱动程序会在底层强制执行L1缓存失效，注意这种失效是在grid之间的。
* .cg: 绕过L1缓存，只在L2缓存
* .cs: Cache Stream，也就是流式数据。这种数据只会访问一两次，因此会在L1和L2 Cache分配采用优先驱逐策略，避免污染缓存
* .lu: Last use，在恢复溢出寄存器和弹出函数堆栈的时候使用，底层的策略跟.cs差不多
* .cv: Cache Volatile，主要用于系统内存(区别于显存)读取，语义大约就是"别信你里面的旧副本！把它标记为失效/丢弃（discard），重新去 PCIe/NVLink 总线对面的 CPU 内存拉一次最新的数据"
st
* .wb: 这是默认的写回缓存操作，语义是“写回时绕过(Bypass)或使当前SM的L1 Cache失效，直接将最新数据写回到L2缓存(或者标记为脏，然后写回显存)，注意无法更新其他SM内部L1上的缓存”。讨论一下`st.wb`和`ld.ca`的缓存一致性问题
  *  SM1 `ld.ca`读全局内存并缓存在L1 ——> SM0执行`st.wb`写全局内存 ——> SM1再次`ld.ca`读，就会命中L1上的脏数据。这里实际上驱动程序会帮我们规避，当一个 Grid（Kernel A）结束，下一个依赖它的 Grid（Kernel B）启动时，GPU 驱动和硬件机制会在两者交接的边界，强制刷新/使全局 L1 缓存行失效（Invalidate L1）。但是跨block的协同操作呢？(同一个kernel)，这个时候就需要 写端：使用内存屏障或者释放语义(fence/release)；读端：Bypass L1(ld.cg)或者硬件原子操作/volatile加载(acquire语义)，才能保证数据可见性
* .cg: 同上，Bypass L1 Cache，仅用L2 Cache
* .cs: 同上，采用优先驱逐策略
* .wt: Cache Write through，缓存直写(到系统内存)
  * st.wb是数据写到L2之后立即返回，L2那行数据标记为脏行，只有这行数据被驱逐或者显示Flush的时候，才会真正写到系统内存。但是st.wt是数据写到L2瞬间，GPU硬件会发起一个跨总线的同步请求，这样CPU就能立即看到GPU写入的最新值。  

# ld 指令
GPU的访存是我们学习的重点，现在学习ld指令，语义是**从内存地址`[a]`加载数据到寄存器`d`**
句法如下
```c++
// 普通的带缓存操作
ld{.weak}{.ss}{.cop}{.level::cache_hint}{.level::prefetch_size}{.vec}.type  d, [a]{.unified}{, cache_policy};
// 带细粒度缓存驱逐优先级的操作
ld{.weak}{.ss}{.level1::eviction_priority}{.level2::eviction_priority}{.level::cache_hint}{.level::prefetch_size}{.vec}.type  d, [a]{.unified}{, cache_policy};
// volatile语义
ld.volatile{.ss}{.level::prefetch_size}{.vec}.type  d, [a];
// relaxed语义
ld.relaxed.scope{.ss}{.level1::eviction_priority}{.level2::eviction_priority}{.level::cache_hint}{.level::prefetch_size}{.vec}.type  d, [a]{, cache_policy};
// acquire语义
ld.acquire.scope{.ss}{.level1::eviction_priority}{.level2::eviction_priority}{.level::cache_hint}{.level::prefetch_size}{.vec}.type  d, [a]{, cache_policy};
// 硬件MMIO语义
ld.mmio.sem.sys{.global}.type  d, [a];

.ss =                       { .const, .global, .local, .param{::entry, ::func}, .shared{::cta, ::cluster} };
.cop =                      { .ca, .cg, .cs, .lu, .cv };
.sem =                      { .acquire, .relaxed };
.level1::eviction_priority = { .L1::evict_normal, .L1::evict_unchanged,
                               .L1::evict_first, .L1::evict_last, .L1::no_allocate };
.level2::eviction_priority = {.L2::evict_normal, .L2::evict_first, .L2::evict_last};
.level::cache_hint =        { .L2::cache_hint };
.level::prefetch_size =     { .L2::64B, .L2::128B, .L2::256B }
.scope =                    { .cta, .cluster, .gpu, .sys };
.vec =                      { .v2, .v4, .v8 };
.type =                     { .b8, .b16, .b32, .b64, .b128,
                              .u8, .u16, .u32, .u64,
                              .s8, .s16, .s32, .s64,
                              .f32, .f64 };
```

来看各修饰符详解
* .ss（State Space，存储空间）
  * .const: 常量内存, readOnly，带constant cache
  * .global: 全局显存
  * .local: 局部显存
  * .param：参数内存（::entry 对应 Kernel 入口参数，::func 对应函数参数）
  * .shared: 共享内存(::cta, ::cluster)，默认是cta
  * 假如不写.ss，默认是通用地址，硬件在运行时动态判定(开销？)
* .cop (Cache Operation，缓存操作策略)
  * .ca: 是默认策略，L1、L2都缓存
  * .cg：Bypass L1，只缓存L2
  * .cs：流式加载，优先驱逐
  * .lu：最后一次使用，基本同.cs
  * .cv: 强行重新拉取，常用于系统内存
* .level1::eviction_priority & .level2::eviction_priority (赋予程序员L1/L2 Cache Line的极致控制权)
  * .evict_normal：默认LRU策略
  * .evict_first：优先驱逐，相当于.cs
  * .evict_last：最后驱逐，高优先级保留
  * .L1::no_allocate：命中就用，没命中也不在L1中分配缓存行
  * .evict_unchange：保持原有的驱逐策略不变
* .scope（Memory Scope，内存作用域）配合`.relaxed`或者`.acquire`语义使用
  * .cta：当前CTA内部可见
  * .cluster：cluster内所有CTA可见
  * .gpu：当前GPU上所有线程可见
  * .sys：全系统可见(包括CPU、PCIe/NVLink设备)
* .level::cache_hint & .level::prefetch_size (缓存提示与预取)
  * .L2::cache_hint：结合 createpolicy 指令传入一个 64 位 Cache Policy 描述符（如指定只留在某一段 L2 Slice）。
  * .prefetch_size (.L2::64B, .128B, .256B)：提示 L2 硬件控制器提前预取（Prefetch） 指定大小的数据到 L2 中，提高后续指令的 Cache Hit 概率。
* .vec (Vector Operations，向量化加载)
  * 支持单条指令同时加载多个连续元素到寄存器组(合并访存的关键)
  * .v2/.v4/.v8
* .type(数据类型)  


上面跟我已知的不一样的是，有这样的描述：
> The .v8 (.vec) qualifier is supported if: .type is .b32 or .s32 or .u32 or .f32
这样的话一条PTX指令支持load总共256bit的数据，有如下限定：
> Support for .level2::eviction_priority qualifier and .v8.b32/.v4.b64 require sm_100 or higher.
很好奇SASS层面，是LDG.256还是两条LDG.128呢？回头测试一下。
```c++
ld.global.L2::evict_last.v8.f32 { %reg0, _, %reg2, %reg3, %reg4, %reg5, %reg6, %reg7}, [addr];
ld.global.L2::evict_last.L1::evict_last.v4.u64 { %reg0, %reg1, %reg2, %reg3}, [addr];
```


## `ld.relaxed`和`ld.acquire`
先来看一下，典型的producer-comsumer模型
```bash
Thread A                  Thread B

store data=123

store flag=1  -------->   load flag

                          load data
```
GPU编译器可能会重排，比如线程A执行顺序可能是
```c++
flag = 1;
data = 123;
```
这样线程B读到的data可能就是旧值。
所以需要memory ordering
PTX提供了下面指令：
```c++
ld.relaxed
ld.acquire

st.relaxed
st.release

atom.relaxed
atom.acquire
atom.release
atom.acq_rel
```

* relaxed 语义是
> 允许最大程度重排，只保证这个load/store操作是原子的
比如`ld.relaxed.gpu.u32 %r1,[addr];`, 意思是：读取 addr 的值，但是不要因为这个load阻止前后的memory操作重排
原子意思是，这条指令是一次完成的，中间不会被其他指令插入
* acquire 语义是
> 当前线程执行 acquire load 后，后面的 memory operation 不能被移动到 acquire load 前面
比如先后执行 `ld.acquire(flag)`和`load data`，一定能保证顺序  
上面的producer-comsumer：
```bash
# thread A
data = 123;
st.release(flag,1);

# thread B
while(ld.acquire(flag)==0);
print(data);
```

# ld.global.nc 指令
使用非一致性缓存加载
前面知道，L1 cache是不保证一致性的，当使用ld.global.nc的时候，就是在向GPU硬件声明：
“保证kernel运行期间，没有任何线程会去写这个地址的数据”
硬件接到这个保证之后，就会：
1. 走只读缓存通道，绕过普通的L1 cache
2. 极高的命中率和极低的读延迟，因为放弃了一致性维护，不需要处理复杂的 Write-back / Invalidation 逻辑，因而能提供极高的读取吞吐量和更低的访问延迟
3. 减少了普通L1/L2的缓存污染，把只读的常量/权重数据路由到只读缓存中，能够为需要频繁读写的变量腾出普通 L1/L2 缓存的空间

简单理解就是跟L1同级别的，有一块只读缓存，所以下面指令可以支持绕过只读缓存
```
ld.global{.cop}.nc{.level::cache_hint}{.level::prefetch_size}.type                 d, [a]{, cache_policy};
ld.global{.cop}.nc{.level::cache_hint}{.level::prefetch_size}.vec.type             d, [a]{, cache_policy};

ld.global.nc{.level1::eviction_priority}{.level2::eviction_priority}{.level::cache_hint}{.level::prefetch_size}.type      d, [a]{, cache_policy};
ld.global.nc{.level1::eviction_priority}{.level2::eviction_priority}{.level::cache_hint}{.level::prefetch_size}.vec.type  d, [a]{, cache_policy};

.cop  =                     { .ca, .cg, .cs };     // cache operation
.level1::eviction_priority = { .L1::evict_normal, .L1::evict_unchanged,
                               .L1::evict_first, .L1::evict_last, .L1::no_allocate};
.level2::eviction_priority = {.L2::evict_normal, .L2::evict_first, .L2::evict_last};
.level::cache_hint =        { .L2::cache_hint };
.level::prefetch_size =     { .L2::64B, .L2::128B, .L2::256B }
.vec  =                     { .v2, .v4, .v8 };
.type =                     { .b8, .b16, .b32, .b64, .b128,
                              .u8, .u16, .u32, .u64,
                              .s8, .s16, .s32, .s64,
                              .f32, .f64 };
```

具体怎么使用呢？
```c++
val = __ldg(ptr);   // 强制走 Read-Only Cache 加载

void kernel(const float* __restrict__ A)    // 编译器自动分析出指针只读且无别名，自动优化为 .nc 加载
```

上面也是我们经常用到的优化手段。