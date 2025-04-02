#pragma once


#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/arch/mma.h"

namespace cutlass {
namespace gemm {
namespace thread {

template<
  ///< shape of gemm problem
  typename Shape_,
  /// Data type of A elements
  typename ElementA_,
  /// Layout of A matrix (concept: layout::MapFunc)
  typename LayoutA_,
  /// Data type of B elements
  typename ElementB_,
  /// Layout of B matrix (concept: layout::MapFunc)
  typename LayoutB_,
  /// Element type of C matrix
  typename ElementC_,
  /// Layout of C matrix (concept: layout::MapFunc)
  typename LayoutC_
>
class MmaGemv {
public:
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;

  /// Data type of operand A
  using ElementA = ElementA_;

  /// Layout of A matrix (concept: layout::MapFunc)
  using LayoutA = LayoutA_;

  /// Data type of operand B
  using ElementB = ElementB_;

  /// Layout of B matrix (concept: layout::MapFunc)
  using LayoutB = LayoutB_;

  /// Element type of operand C
  using ElementC = ElementC_;

  /// Layout of C matrix (concept: layout::MapFunc)
  using LayoutC = LayoutC_;

  /// Underlying mathematical operator
  using Operator = arch::OpMultiplyAdd;

  using FragmentA = cutlass::Array<ElementA, Shape::kMK>;

  using FragmentB = cutlass::Array<ElementB, Shape::kMK>;

  using FragmentC = cutlass::Array<ElementC, Shape::kM>;

  using ArchMmaOperator = cutlass::arch::Mma<gemm::GemmShape<1, 1, Shape::kK>,
                                             1,
                                             ElementA,
                                             LayoutA,
                                             ElementB,
                                             LayoutB,
                                             ElementC,
                                             LayoutC,
                                             Operator>;

 public:
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC & D,
    FragmentA const & A,
    FragmentB const & B,
    FragmentC const & C) {

    D = C;

    ArchMmaOperator mma_op;

    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < Shape::kM; ++m) {
      cutlass::Array<ElementC, 1> d;
      cutlass::Array<ElementA, 1> a;
      cutlass::Array<ElementB, 1> b;

      *d.data() = *(D.data() + m);
      *a.data() = *(A.data() + m * Shape::kK);
      *b.data() = *(B.data() + m * Shape::kK);

      mma_op(d, a, b, d);
      *(D.data() + m) = *d.data();
    }

  }

};

} /// end of namespace thread
} /// end of namespace gemm
} /// end of namespace cutlass