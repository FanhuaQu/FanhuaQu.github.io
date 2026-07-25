# ldmatrix
## 指令介绍
ldmatrix作用是把数据从共享内存加载到寄存器，主要配合mma指令使用
是一个warp级别的指令，一个warp中的32个线程合作搬运数据
```c++
ldmatrix.sync.aligned.shape.num{.trans}{.ss}.type r, [p];

ldmatrix.sync.aligned.m8n16.num{.ss}.dst_fmt.src_fmt        r, [p];
ldmatrix.sync.aligned.m16n16.num.trans{.ss}.dst_fmt.src_fmt r, [p];

.shape   = {.m8n8, .m16n16};
.num     = {.x1, .x2, .x4};
.ss      = {.shared{::cta}};
.type    = {.b16, .b8};
.dst_fmt = { .b8x16 };
.src_fmt = { .b6x16_p32, .b4x16_p64 };
```

从地址操作数`p`指示的位置，将一个或多个矩阵从共享内存加载到目标寄存器`r`中

* .sync 说明这个指令是同步的，直到warp内的所有线程执行相同的ldmatrix指令后才恢复执行
* .aligned 说明这个指令是warp级别的，也即warp内的32个线程在控制流上是对齐的，不存在条件分化
* .shape 限定符指示加载数据的维度，不同的数据类型支持不同的shape，包括`m8n8`、`m16n16`和`m8n16`

| shape   | Matrix shape | Elements size    |
| ------- | ------------ | ---------------- |
| .m8n8   | 8x8          | 16bit            |
| .m16n16 | 16x16        | 8bit、6bit、4bit |
| .m8n16  | 8x16         | 6bit、4bit       |

* .type里面的b16代表16bit，因此可以处理8x8大小的bf16/fp16数据，也可也处理8x4大小的float数据
* 目标操作数`r`是一个向量表达式，根据`.num`的选值，由1、2或4个32位寄存器组成
* .trans表示加载中是否进行转置

8x8大小的bf16数据是ldmatrix加载的基本单位。因为8x8的bf16数据一共有128bytes，在共享内存加载中属于一个内存事务，在没有bank conflict情况下可以在一个时钟周期内完成。

> .m16n16和.m8n16 shape只在 sm_90以上版本支持(包括sm120); .b8以及.src_fmt、.dst_fmt 两个限定符同样只支持sm90以上架构

## 使用方法
ldmatrix加载数据的过程可以视为两部分，第一部分，每个线程从传入地址中加载128bit的数据。第二部分，线程之间相互分发数据，得到每个线程最终对应的数据。

因为加载的矩阵块大小是8x8的，每行数据连续，因此只需要计算8个首地址，假设第i行的首地址是addr_i，线程和地址的对应关系如下
| .num | Thread 0~7    | Thread 8~15    | Thread 16~23    | Thread 24~31    |
| ---- | ------------- | -------------- | --------------- | --------------- |
| .x1  | addr_0~addr_7 | -              | -               | -               |
| .x2  | addr_0~addr_7 | addr_8~addr_15 | -               | -               |
| .x4  | addr_0~addr_7 | addr_8~addr_15 | addr_16~addr_23 | addr_24~addr_31 |
* num是`.x2`的时候，加载了两次8x8矩阵，也就是16x8，`.x4`的时候就是4个8x8。  

当num=.x1的时候，加载完成之后，线程和数据对应关系如下:
| Row\Col | 0      | 1      | 2      | 3      | 4      | 5      | 6      | 7      |
| ------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| 0(T0)   | T0:V0  | T0:V1  | T1:V0  | T1:V1  | T2:V0  | T2:V1  | T3:V0  | T3:V1  |
| 1(T1)   | T4:V0  | T4:V1  | T5:V0  | T5:V1  | T6:V0  | T6:V1  | T7:V0  | T7:V1  |
| 2(T2)   | T8:V0  | T8:V1  | T9:V0  | T9:V1  | T10:V0 | T10:V1 | T11:V0 | T11:V1 |
| 3(T3)   | T12:V0 | T12:V1 | T13:V0 | T13:V1 | T14:V0 | T14:V1 | T15:V0 | T15:V1 |
| 4(T4)   | T16:V0 | T16:V1 | T17:V0 | T17:V1 | T18:V0 | T18:V1 | T19:V0 | T19:V1 |
| 5(T5)   | T20:V0 | T20:V1 | T21:V0 | T21:V1 | T22:V0 | T22:V1 | T23:V0 | T23:V1 |
| 6(T6)   | T24:V0 | T24:V1 | T25:V0 | T25:V1 | T26:V0 | T26:V1 | T27:V0 | T27:V1 |
| 7(T7)   | T28:V0 | T28:V1 | T29:V0 | T29:V1 | T30:V0 | T30:V1 | T31:V0 | T31:V1 |

当使用可选的限定符`.trans`的时候，线程与元素的对应关系就是上面矩阵的转置。

对于m16n16和m8n16也是类似的，由于都是低精度的，比如fp8，在m8n16时，每个线程用一个.b32的目标寄存器存储连续的4个8bit元素(6bit和4bit填充到8bit)，所以还是4个线程负责一整行


## 测试代码：
* .x1  
实际加载数据的是前8个线程
```c++
template <typename T>
__device__ __inline__ uint32_t cast_smem_ptr_to_uint(T *smem_ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
};

template<class T>
__global__ void ldmatrix_x1(T* S, T* D, int M, int N)
{
    int tid = threadIdx.x;

    __shared__ T smem_src[64];
    smem_src[tid] = S[tid];
    smem_src[tid+32] = S[tid+32];
    __syncthreads();

    T* smem_addr = nullptr;
    // 前8个线程提供每行的起始地址
    if(tid < 8){
        smem_addr = smem_src + tid * N;
    }

    uint32_t dst1 ;
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_addr);   // 将地址转化成ptx需要的uint32_t类型
    // 注意PTX语法，{}里面是寄存器列表，[]表示内存地址
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                : "=r"(dst1)
                : "r"(smem_int_ptr)
            );
    
    int r = tid / 4;
    int c1 = (tid % 4) * 2;
    int c2 = c1 + 1;

    // 解包32位寄存器为两个16位的T类型数据
    T* dst_val = reinterpret_cast<T*>(&dst1);

    D[r*N + c1] = dst_val[0];
    D[r*N + c2] = dst_val[1];
}
```

* .x2  
需要一个warp的前16个线程加载数据
```c++
template<class T>
__global__ void ldmatrix_x2(T* S, T* D, int M, int N)
{
    int tid = threadIdx.x;

    __shared__ T smem_src[128];     // 16 x 8
    smem_src[tid] = S[tid];
    smem_src[tid+32] = S[tid+32];
    smem_src[tid+64] = S[tid+64];
    smem_src[tid+96] = S[tid+96];

    __syncthreads();

    T* smem_addr = nullptr;

    if(tid < 16){
        smem_addr = smem_src + tid * N;
    }

    uint32_t dst1, dst2;
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_addr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                : "=r"(dst1), "=r"(dst2)
                : "r"(smem_int_ptr)
            );
    
    int r = tid / 4;
    int c1 = (tid % 4) * 2;
    int c2 = c1 + 1;

    T* dst_val1 = reinterpret_cast<T*>(&dst1);
    T* dst_val2 = reinterpret_cast<T*>(&dst2);

    D[r*N + c1] = dst_val1[0];
    D[r*N + c2] = dst_val1[1];
    D[(r+8)*N + c1] = dst_val2[0];
    D[(r+8)*N + c2] = dst_val2[1];
}
```

* .x4  
一个warp的32个线程一起搬运数据
前面提到ldmatrix的基本单位是8x8的块，因此这里的.x4可以搬运32x8的块，也可以搬运16x16的块，具体取决于代码里面smem_addr的计算方法。
```c++

template<class T>
__global__ void ldmatrix_x4(T* S, T* D, int M, int N)
{
    int tid = threadIdx.x;

    __shared__ T smem_src[256];     // 32 x 8
    for(int i = tid; i < M*N; i += 32){
        smem_src[i] = S[i];
    }

    __syncthreads();

    T* smem_addr = nullptr;

    if(tid < 32){
        smem_addr = smem_src + tid * N;
    }

    uint32_t dst1, dst2, dst3, dst4;
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_addr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(dst1), "=r"(dst2), "=r"(dst3), "=r"(dst4)
                : "r"(smem_int_ptr)
            );
    
    int r = tid / 4;
    int c1 = (tid % 4) * 2;
    int c2 = c1 + 1;

    T* dst_val1 = reinterpret_cast<T*>(&dst1);
    T* dst_val2 = reinterpret_cast<T*>(&dst2);
    T* dst_val3 = reinterpret_cast<T*>(&dst3);
    T* dst_val4 = reinterpret_cast<T*>(&dst4);

    D[r*N + c1] = dst_val1[0];
    D[r*N + c2] = dst_val1[1];
    D[(r+8)*N + c1] = dst_val2[0];
    D[(r+8)*N + c2] = dst_val2[1];
    D[(r+16)*N + c1] = dst_val3[0];
    D[(r+16)*N + c2] = dst_val3[1];
    D[(r+24)*N + c1] = dst_val4[0];
    D[(r+24)*N + c2] = dst_val4[1];
}
```

这里会有一个疑问，8x8的时候，只有前8个线程计算了smem的地址，那最后是怎么做到每个线程都得拿到数据的呢？  
这里是因为ldmatrix一个线程从smem直接加载128bit的数据，然后在warp内将数据shuffle到每个线程需要的位置(合并成一条指令了，所以也不会有多占用寄存器的问题)。这样就大幅减小了代码的复杂性，也能避免bank conflict。


# stmatrix
将一个或者多个矩阵集中存储到共享内存中
```c++
stmatrix.sync.aligned.shape.num{.trans}{.ss}.type [p], r;

.shape  = {.m8n8, .m16n8};
.num    = {.x1, .x2, .x4};
.ss     = {.shared{::cta}};
.type   = {.b16, .b8};
```
基本用法是和ldmatrix指令一样的
> stmatrix指令需要sm_90及以上的版本
> .m16n8、.b8只支持sm_90以上版本

## 代码测试
* .x1
```c++
template<class T>
__global__ void stmatrix_x1(T* S, T* D, int M, int N)
{
    int tid = threadIdx.x;

    __shared__ T smem_src[64];
    __shared__ T smem_dst[64];
    smem_src[tid] = S[tid];
    smem_src[tid+32] = S[tid+32];

    __syncthreads();

    T* smem_addr = nullptr, smem_addr_dst = nullptr;

    if(tid < 8){
        smem_addr = smem_src + tid * N;
        smem_addr_dst = smem_dst + tid * N;
    }

    uint32_t dst1 ;
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_addr);
    uint32_t smem_int_ptr_dst = cast_smem_ptr_to_uint(smem_addr_dst);
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                : "=r"(dst1)
                : "r"(smem_int_ptr)
            );
    
    asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                : "r"(dst1)
                : "r"(smem_int_ptr_dst)
            );
    
    __syncthreads();

    int idx = tid * 2;
    D[idx] = smem_dst[idx];
    D[idx+1] = smem_dst[idx+1];
}
```

# 总结
ldmatrix和stmatrix指令用于smem和寄存器之间进行数据传输，常和mma指令一起使用，由于mma指令需要的寄存器布局比较复杂，手动计算偏移容易出错，还要考虑bank conflict的问题。ldmatrix指令则完美解决了这一问题，一个warp可以完成一块的加载。  
