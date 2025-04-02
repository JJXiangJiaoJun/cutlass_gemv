#pragma once

#include "cutlass/layout/matrix.h"
#include "cutlass/epilogue/thread/activation.h"

namespace reference {

template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementOutput,
  typename LayoutOutput,
  typename ElementAccumulate = float,
  typename ElementCompute = ElementAccumulate,
  typename Activation = cutlass::epilogue::thread::Identity<ElementCompute>
>
struct Gemv;

template <
  typename ElementA_,
  typename ElementB_,
  typename ElementOutput_,
  typename ElementAccumulate_,
  typename ElementCompute_,
  typename Activation_
>
struct Gemv<
 ElementA_,
 cutlass::layout::RowMajor,
 ElementB_,
 cutlass::layout::ColumnMajor,
 ElementOutput_,
 cutlass::layout::RowMajor,
 ElementAccumulate_,
 ElementCompute_,
 Activation_
> {
  using ElementA = ElementA_;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = ElementB_;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementOutput = ElementOutput_;
  using LayoutOutput = cutlass::layout::RowMajor;
  using ElementAccumulate = ElementAccumulate_;
  using ElementCompute = ElementCompute_;
  using Activation = Activation_;

  Activation act;

  void operator()(const ElementA *A,
                  const ElementB *B,
                  const ElementOutput *C,
                  ElementOutput *D,
                  int M,
                  int N,
                  int K,
                  ElementAccumulate alpha,
                  ElementAccumulate beta) {
    for (int m_i = 0; m_i < M; ++m_i) {
      ElementAccumulate tmp =
          (C == nullptr ? ElementAccumulate(0) : (static_cast<ElementAccumulate>(C[m_i])));

      for (int k_i = 0; k_i < K; ++k_i) {
        tmp += alpha * ElementAccumulate(A[m_i * K + k_i]) * ElementAccumulate(B[k_i]);
      }

      ElementCompute res = static_cast<ElementCompute>(tmp);
      res = act(res);
      D[m_i] = static_cast<ElementOutput>(res);

    }
  }
};

template <
  typename ElementA_,
  typename ElementB_,
  typename ElementOutput_,
  typename ElementAccumulate_,
  typename ElementCompute_,
  typename Activation_
>
struct Gemv<
 ElementA_,
 cutlass::layout::ColumnMajor,
 ElementB_,
 cutlass::layout::RowMajor,
 ElementOutput_,
 cutlass::layout::ColumnMajor,
 ElementAccumulate_,
 ElementCompute_,
 Activation_
> {
  using ElementA = ElementA_;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = ElementB_;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementOutput = ElementOutput_;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  using ElementAccumulate = ElementAccumulate_;
  using ElementCompute = ElementCompute_;
  using Activation = Activation_;

  Activation act;

  void operator()(const ElementA *A,
                  const ElementB *B,
                  const ElementOutput *C,
                  ElementOutput *D,
                  int M,
                  int N,
                  int K,
                  ElementAccumulate alpha,
                  ElementAccumulate beta) {
    for (int m_i = 0; m_i < M; ++m_i) {
      ElementAccumulate tmp =
          (C == nullptr ? ElementAccumulate(0) : (static_cast<ElementAccumulate>(C[m_i])));

      for (int k_i = 0; k_i < K; ++k_i) {
        tmp += alpha * ElementAccumulate(A[m_i + k_i * M]) * ElementAccumulate(B[k_i]);
      }

      ElementCompute res = static_cast<ElementCompute>(tmp);
      res = act(res);
      D[m_i] = static_cast<ElementOutput>(res);

    }
  }
};

} /// end of namespace reference