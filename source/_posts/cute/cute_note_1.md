工作一年多，已经接触了很多cuda算子，也写过一些基于cute的但是理解一直不深，借此机会整理一番，加深自己的理解。

note 1先整理一下Layout

# Layout

对于一个矩阵，逻辑上可能是[3][4]，也可能是[3][5][4]，但是在显存中是连续存储的，GPU看到的是`address = base + offset`  

以二维矩阵为例，A[M][N]  
如果是行主序，那么`A[i][j] = i * N + j`
这里将逻辑坐标`Coordinate`和物理地址映射起来的，是一个函数。Layout本质上就是这个函数的抽象

有了Layout，我们就能完成下面的映射
```bash
Coord (i, j)
    |
    |   Layout
    ↓
offset
    |
    |   + base point
    ↓
address
```
来看Layout的定义
```c++
template <class Shape, class Stride = LayoutLeft::Apply<Shape> >
struct Layout
    : private cute::tuple<Shape, Stride>   // EBO for static layouts
{
  Layout(Shape  const& shape  = {}, Stride const& stride = {})
      : cute::tuple<Shape, Stride>(shape, stride)
  {}

}
```

可以看到默认是列主序的(LayoutLeft)

源码里面涉及到很多方法，这里不打算深究。个人觉得花大量的时间扎进去搞懂各种变换的数学推导过程是不值得的。我要的是，对于任何一个变换，脑子里都有一幅变换的letax即可

## CuTe Layout 的递归结构与 Layout Algebra
GPU本身是分层次结构的，从上往下是 `GPU —— Cluster —— CTA —— Warp-group —— Warp —— lane`
某个线程的描述可能是`(cluster_id, warp_id, lane_id)`
Layout的嵌套可以用于描述这种复杂的层次关系

> Layout 可以进行代数运算
这是Layout最强的地方，也是最抽象的部分。
对于GPU开发来说，无非就是每个线程处理哪部分数据。
这里就会涉及到
1. 线程的布局 L1
2. 数据的布局 L2
那么描述某个线程对应哪个数据的映射关系，就可以用下面式子表示
> L(Thread) = L2(L1(Thread))

### composition
composition(A, B)
> 创建一个新的Layout，使B的坐标经过A映射
前面提到过，Layout其实就是一个函数，所以跟函数的嵌套一样
composition(L_A, L_B)表示的Layout就是L_A(L_B(Thread))  

举一个具体的例子
大矩阵global_memory: A[128][128]
每个CTA处理32 x 32的Tile
那么就有两层Layout
首先是gmem的layout
```c++
auto gmem_layout = make_layout(make_shape(128, 128), make_stride(128, 1));
```
也就是行主序，映射关系是`G(m, n) = 128m + n`
那怎么描述某个tile在整个矩阵的哪个位置呢？
比如tile_id = (2, 1)
对应的gmem的起点：m0 = 2x32 = 64; n0 = 1x32 = 32
所以tile内的坐标(i,j)，就映射到了global的坐标(64+i, 32+j)
这就是第一个Layout，描述的是Tile内元素的逻辑坐标到Global坐标的映射关系。

第二个layout描述了Global Coordinate到memory offset的映射关系
在这里就是
```c++
composition(
    gmem_layout,
    tile_layout
)
```

```c++
template <class LShape, class LStride,
          class RShape, class RStride>
CUTE_HOST_DEVICE constexpr
auto
composition(Layout<LShape,LStride> const& lhs,
            Layout<RShape,RStride> const& rhs)
{
  auto flat_lhs = detail::coalesce_x(lhs, coprofile(rhs));
  return detail::composition_impl(flat_lhs.shape(), flat_lhs.stride(), rhs.shape(), rhs.stride());
}
```

找个更直观的例子
gmem A[8][8]  (8,8):(8,1)
对应的映射关系就是offset_(m,n) = 8m+n
假如一个warp有4个线程，线程的layout是4:1，也就是行方向排布
那么现在知道thread_id，怎么知道对应的gmem地址呢？
路径就是thread_id ——> thread_layout  ——> row coordinate ——> gmem_layout ——> address
对应的代码就是
```c++
auto thread_to_gmem =
    composition(
        gmem_layout,
        thread_layout
    );
```
返回的是一个新的Layout，对应的映射是thread_id ——> gmem_offset

