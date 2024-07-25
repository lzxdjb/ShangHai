## 概述

1. **将数据从 q 和 k 复制到 GPU（HtoD）**：将数据从Host传输到 Device.
2. **在内核中执行笛卡尔积、编码和存储**：在内核中执行笛卡尔积、编码、查找 LUT 和存储结果。
3. **将数据复制到存储（DtoH）**：将计算结果从 Device 传回Host。

## 执行
mkdir build
cd build
cmake ..
make
cd example
./Combined


## 实验结果
Combined 中包含了我写四个版本
type.cuh 可以修改参数

**Results:**

`ql = 4 ; nh = 4`

- Time for sequential transfer and execute (ms): 73.896317  
- Time for stream execute (ms): 0.393440  
- Time for MoreFancy execute (ms): 0.424800  
- Time for Ultra execute (ms): 0.417376  

`ql = 8 ; nh = 8`

- Time for sequential transfer and execute (ms): 211.430496  
- Time for stream execute (ms): 99.160835  
- Time for MoreFancy execute (ms): 210.150589  
- Time for Ultra execute (ms): 0.956000  

`ql = 16 ; nh = 16`

- Time for sequential transfer and execute (ms): 790.458130  
- Time for stream execute (ms): 590.692383  
- Time for MoreFancy execute (ms): 825.458679  
- Time for Ultra execute (ms): 2.335008  

`ql = 32 ; nh = 32`

- Time for sequential transfer and execute (ms): 3732.801758  
- Time for stream execute (ms): 3217.209229  
- Time for MoreFancy execute (ms): 3949.333740  
- Time for Ultra execute (ms): 7.954624    

<img src="./picture/FinalResult.png" alt="图片描述" width="400" />

我没有在往上增加ql,nh，因为我的 GPU 很糟糕。

## 具体思想请见实验报告!
