# cutlass_gemv
Gemv implementation with cutlass

# Quick Start

```shell
git clone git@github.com:JJXiangJiaoJun/cutlass_gemv.git
cd cutlass_gemv
git clone git@github.com:NVIDIA/cutlass.git
cd cutlass && git checkout v3.4.1 && cd ..
nvcc --std=c++17 -arch=sm_86 --expt-relaxed-constexpr -O2 -I ./ -I ./cutlass/include gemv_test.cu -o gemv_test
```