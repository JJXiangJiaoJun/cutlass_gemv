#include <iostream>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/array.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/epilogue/thread/activation.h"

#include "reference/gemv.h"
#include "reference/initializer.h"

#include "gemv/device/gemv_adaptor.h"

using ElementA = float;
using LayoutA  = cutlass::layout::RowMajor;
using ElementB = float;
using LayoutB  = cutlass::layout::ColumnMajor;
using ElementC = float;
using LayoutC  = cutlass::layout::RowMajor;
using ElementAccumulator = float;
using OperatorClass = cutlass::arch::OpClassSimt;
using ArchTag = cutlass::arch::Sm50;
using ThreadBlockShape = cutlass::gemm::GemmShape<4, 1, 256>;
using WarpShape = cutlass::gemm::GemmShape<4, 1, 64>;
using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
using WarpThreadArrangement = cutlass::MatrixShape<2, 16>;
using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,            // <- data type of output matrix
    1,                   // <- this is the number of elements per
                         // vectorized memory access. For half
                         // precision, it's 8 elements. This becomes
                         // the vector width of math instructions in
                         // epilogue too
    ElementAccumulator,  // <- data type of accumulator
    ElementAccumulator,  // <- data type for alpha in linear combination function
    cutlass::epilogue::thread::ScaleType::NoBetaScaling>;  // <- alpha x C + bias

static const int kGemmN = 1;
static const int kStages = 1;
static const int kAlignmentA = 4;
static const int kAlignmentB = 4;


using DeviceKernel = cutlass::gemm::device::GemvAdaptor<ElementA,
                                                        LayoutA,
                                                        ElementB,
                                                        LayoutB,
                                                        ElementC,
                                                        LayoutC,
                                                        ElementAccumulator,
                                                        OperatorClass,
                                                        ArchTag,
                                                        ThreadBlockShape,
                                                        WarpShape,
                                                        InstructionShape,
                                                        WarpThreadArrangement,
                                                        EpilogueOutputOp,
                                                        kStages,
                                                        kAlignmentA,
                                                        kAlignmentB>;

using HostKernel = reference::Gemv<ElementA, LayoutA, ElementB, LayoutB,
                                   ElementC, LayoutC, ElementAccumulator>;

void device_gemv(const ElementA *ptr_A,
                 const ElementB *ptr_B,
                 const ElementC *ptr_C,
                 ElementC *ptr_D,
                 int m,
                 int n,
                 int k,
                 float alpha = 1.0f) {
  typename DeviceKernel::Arguments args {
    cutlass::make_Coord(m, n, k),
    cutlass::make_TensorRef(const_cast<ElementA *>(ptr_A), cutlass::layout::RowMajor(k)),
    cutlass::make_TensorRef(const_cast<ElementB *>(ptr_B), cutlass::layout::ColumnMajor(0)),
    cutlass::make_TensorRef(const_cast<ElementC *>(ptr_C), cutlass::layout::RowMajor(1)),
    cutlass::make_TensorRef(const_cast<ElementC *>(ptr_D), cutlass::layout::RowMajor(1)),
    {alpha}
  };

  DeviceKernel op;
  op.initialize(args);
  op.run();
}

void host_gemv(const ElementA *ptr_A,
               const ElementB *ptr_B,
               const ElementC *ptr_C,
               ElementC *ptr_D,
               int m,
               int n,
               int k,
               float alpha = 1.0f,
               float beta = 0.0f) {
  HostKernel host_op;
  host_op(ptr_A, ptr_B, ptr_C, ptr_D, m, n, k, alpha, beta);
}

int main() {

  int M = 1024, K = 4096;

  float alpha = 1.0f;

  ElementA *h_A = new ElementA[M * K];
  ElementB *h_B = new ElementB[kGemmN * K];
  ElementC *h_bias = new ElementC[M];
  ElementC *h_D = new ElementC[M * kGemmN];

  ElementC *result_D = new ElementC[M * kGemmN];

  reference::random_initializer<ElementA>::init(h_A, M * K);
  reference::random_initializer<ElementB>::init(h_B, kGemmN * K);
  reference::random_initializer<ElementC>::init(h_bias, M);

  ElementA *d_A;
  ElementB *d_B;
  ElementC *d_bias;
  ElementC *d_D;

  cudaMalloc(&d_A, M * K * sizeof(ElementA));
  cudaMalloc(&d_B, kGemmN * K * sizeof(ElementB));
  cudaMalloc(&d_bias, M * sizeof(ElementC));
  cudaMalloc(&d_D, M * kGemmN * sizeof(ElementC));

  cudaMemcpy(d_A, h_A, M * K * sizeof(ElementA), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, kGemmN * K * sizeof(ElementB), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias, h_bias, M * sizeof(ElementC), cudaMemcpyHostToDevice);


  // for (int i = 0; i < 10; i++)
  device_gemv(d_A, d_B, d_bias, d_D, M, kGemmN, K, alpha);

  cudaMemcpy(result_D, d_D, M * kGemmN * sizeof(ElementC), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  cudaError_t err = cudaGetLastError();

  if (err != cudaSuccess) {
    std::cout << "err" << std::endl;
    exit(-1);
  }

#ifdef HOST_CHECK
  host_gemv(h_A, h_B, h_bias, h_D, M, kGemmN, K, alpha);

  for (int m = 0; m < M; m++) {
    float abs_err = fabs(float(h_D[m]) - float(result_D[m]));
    if (abs_err > 2e-4) {
      std::cout <<"m: " << m << " cpu: " << float(h_D[m]) << "\tgpu: " << float(result_D[m]) << "\tdiff: " << abs_err << std::endl;
    }
  }
#endif



  delete[] h_A;
  delete[] h_B;
  delete[] h_bias;
  delete[] result_D;
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_bias);
  cudaFree(d_D);

  return 0;
}