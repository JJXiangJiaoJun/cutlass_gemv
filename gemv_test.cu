#include <iostream>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/half.h"
#include "cutlass/arch/arch.h"
#include "cutlass/array.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/epilogue/thread/activation.h"

#include "reference/gemv.h"
#include "reference/initializer.h"

#include "gemv/device/gemv_adaptor.h"

using ElementA = cutlass::half_t;
using LayoutA  = cutlass::layout::RowMajor;
using ElementB = ElementA;
using LayoutB  = cutlass::layout::ColumnMajor;
using ElementC = cutlass::half_t;
using LayoutC  = cutlass::layout::RowMajor;
using ElementAccumulator = float;
using OperatorClass = cutlass::arch::OpClassSimt;
using ArchTag = cutlass::arch::Sm50;
using ThreadBlockShape = cutlass::gemm::GemmShape<4, 1, 512>;
using WarpShape = cutlass::gemm::GemmShape<2, 1, 256>;
using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
using WarpThreadArrangement = cutlass::MatrixShape<2, 16>;

///< ====================================================================
///< Test for column major A matrix
// using ElementA = cutlass::half_t;
// using LayoutA  = cutlass::layout::ColumnMajor;
// using ElementB = ElementA;
// using LayoutB  = cutlass::layout::RowMajor;
// using ElementC = cutlass::half_t;
// using LayoutC  = cutlass::layout::ColumnMajor;
// using ElementAccumulator = float;
// using OperatorClass = cutlass::arch::OpClassSimt;
// using ArchTag = cutlass::arch::Sm50;
// using ThreadBlockShape = cutlass::gemm::GemmShape<128, 1, 16>;
// using WarpShape = cutlass::gemm::GemmShape<64, 1, 8>;
// using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
// using WarpThreadArrangement = cutlass::MatrixShape<8, 4>;


static const int kGemmN = 1;
static const int kStages = 1;
static const int kAlignmentA = 16 / sizeof(ElementA);
static const int kAlignmentB = std::is_same_v<LayoutB, cutlass::layout::RowMajor> ? 1 : (16 / sizeof(ElementB));
static const int kAlignmentC = std::is_same_v<LayoutC, cutlass::layout::RowMajor> ? 1 : kAlignmentA;

using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,            // <- data type of output matrix
    kAlignmentC,         // <- this is the number of elements per
                         // vectorized memory access. For half
                         // precision, it's 8 elements. This becomes
                         // the vector width of math instructions in
                         // epilogue too
    ElementAccumulator,  // <- data type of accumulator
    ElementAccumulator,  // <- data type for alpha in linear combination function
    cutlass::epilogue::thread::ScaleType::NoBetaScaling>;  // <- alpha x C + bias


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

using HostKernel =
    reference::Gemv<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
                    ElementAccumulator, ElementAccumulator,
                    cutlass::epilogue::thread::Identity<ElementAccumulator>>;

void device_gemv(const ElementA *ptr_A,
                 const ElementB *ptr_B,
                 const ElementC *ptr_C,
                 ElementC *ptr_D,
                 int m,
                 int n,
                 int k,
                 float alpha = 1.0f) {

  int lda = std::is_same_v<LayoutA, cutlass::layout::RowMajor> ? k : m;
  int ldb = std::is_same_v<LayoutB, cutlass::layout::ColumnMajor> ? 0 : 1;
  int ldc = std::is_same_v<LayoutC, cutlass::layout::RowMajor> ? 1 : m;
  int ldd = ldc;

  typename DeviceKernel::Arguments args{
    cutlass::make_Coord(m, n, k),
    cutlass::make_TensorRef(ptr_A, LayoutA(lda)),
    cutlass::make_TensorRef(ptr_B, LayoutB(ldb)),
    cutlass::make_TensorRef(ptr_C, LayoutC(ldc)),
    cutlass::make_TensorRef(ptr_D, LayoutC(ldd)),
    {static_cast<ElementAccumulator>(alpha)}
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
               float alpha = 1.0f) {
  HostKernel host_op;
  host_op(ptr_A, ptr_B, ptr_C, ptr_D, m, n, k, static_cast<ElementAccumulator>(alpha));
}

int main() {

  int M = 2048, K = 4096;

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

  cudaError_t result = cudaGetLastError();

  if (result != cudaSuccess) {
    std::cout << "Execution error: " << cudaGetErrorString(result) << std::endl;
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