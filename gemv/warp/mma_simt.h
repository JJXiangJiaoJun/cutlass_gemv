#pragma once


#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "gemv/thread/mma.h"

#include "gemv/warp/mma_simt_policy.h"
#include "gemv/warp/mma_simt_tile_iterator.h"

namespace cutlass {
namespace gemm {
namespace warp {

template<
  ///< Size of Gemm Problem shape
  typename Shape_,
  typename RegisterTileShape_,
  typename ElementA_,
  typename LayoutA_,
  typename ElementB_,
  typename LayoutB_,
  typename ElementC_,
  typename LayoutC_,
  typename Policy_
>
class MmaSimtGemv {
public:
  /// Shape of warp-level matrix operation (concept: GemmShape)
  using Shape = Shape_;

  using RegisterTileShape = RegisterTileShape_;

  /// Data type of multiplicand A
  using ElementA = ElementA_;

  /// Layout of multiplicand A
  using LayoutA = LayoutA_;

  /// Data type of multiplicand B
  using ElementB = ElementB_;

  /// Layout of multiplicand B
  using LayoutB = LayoutB_;

  /// Data type of accumulator matrix C
  using ElementC = ElementC_;

  /// Layout of accumulator matrix C
  using LayoutC = LayoutC_;

  /// Shape of the warp in units of thread (concept: MmaLanePolicySimt)
  using Policy = Policy_;

  static const bool use_dp4a = false;

  /// Shape of underlying instruction
  using ThreadMma = thread::MmaGemv<
      gemm::GemmShape<Shape::kM / Policy::WarpShape::kRow, 1, use_dp4a ? 4 : 1>,
      ElementA,
      LayoutA,
      ElementB,
      LayoutB,
      ElementC,
      LayoutC>;

  /// Underlying matrix multiply operator (concept: arch::Mma)
  using ArchMmaOperator = typename ThreadMma::ArchMmaOperator;
  using InstructionShape = typename ArchMmaOperator::Shape;
  using WarpMmaShape = MatrixShape<Shape::kM, InstructionShape::kK>;

public:
  ///<
  using IteratorA = MmaSimtRegisterTileIterator<WarpMmaShape, RegisterTileShape, Operand::kA, ElementA, Policy, LayoutA>;
  using FragmentA = typename IteratorA::Fragment;

  using IteratorB = MmaSimtRegisterTileIterator<WarpMmaShape, RegisterTileShape, Operand::kB, ElementB, Policy, LayoutB>;
  using FragmentB = typename IteratorB::Fragment;

  using FragmentC = typename ThreadMma::FragmentC;


public:
   CUTLASS_DEVICE
   MmaSimtGemv() {}

   CUTLASS_DEVICE
   void operator()(FragmentC &d, FragmentA a, FragmentB b, FragmentC const &c) {
    ThreadMma mma;
    mma(d, a, b, c);
   }
};

} /// end of namespace warp
} /// end of namespace gemm
} /// end of namespace cutlass