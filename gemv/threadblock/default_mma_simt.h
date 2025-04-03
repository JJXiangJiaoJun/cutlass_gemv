#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"

#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/arch/mma.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

#include "gemv/threadblock/default_mma.h"
#include "gemv/threadblock/default_mma_core_simt.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

template<
  /// Element type for A matrix operand
  typename ElementA_,
  /// Access granularity of A matrix in units of elements
  int AlignmentA,
  /// Element type for B matrix operand
  typename ElementB_,
  /// Access granularity of B matrix in units of elements
  int AlignmentB,
  /// Element type for internal accumulation
  typename ElementAccumulator_,
  /// Tag indicating architecture to tune for
  typename ArchTag_,
  /// Threadblock-level tile size (concept: GemmShape)
  typename ThreadblockShape_,
  /// Warp-level tile size (concept: GemmShape)
  typename WarpShape_,
  /// Warp thread layout (concept: MatrixShape)
  typename WarpThreadArrangement_
>
struct DefaultMmaGemv<
 ElementA_,
 cutlass::layout::RowMajor,
 AlignmentA,
 ElementB_,
 cutlass::layout::ColumnMajor,
 AlignmentB,
 ElementAccumulator_,
 ArchTag_,
 ThreadblockShape_,
 WarpShape_,
 GemmShape<1, 1, 1>,
 WarpThreadArrangement_,
 arch::OpClassSimt
>{

  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using ElementAccumulator = ElementAccumulator_;
  using ArchTag = ArchTag_;

  using ThreadBlockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = GemmShape<1, 1, 1>;
  using WarpThreadArrangement = WarpThreadArrangement_;

  static const int kAlignmentA = AlignmentA;
  static const int kAlignmentB = AlignmentB;

  using MmaCore =
      typename cutlass::gemm::threadblock::DefaultMmaCoreGemv<ThreadBlockShape,
                                                              WarpShape,
                                                              InstructionShape,
                                                              WarpThreadArrangement,
                                                              ElementA,
                                                              cutlass::layout::RowMajor,
                                                              ElementB,
                                                              cutlass::layout::ColumnMajor,
                                                              ElementAccumulator,
                                                              cutlass::layout::RowMajor,
                                                              arch::OpClassSimt>;

  using IteratorA =
      cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
                                                              ElementA,
                                                              cutlass::layout::RowMajor,
                                                              1,
                                                              typename MmaCore::IteratorThreadMapA,
                                                              kAlignmentA>;

  using IteratorB =
      cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kM>,
                                                              ElementB,
                                                              cutlass::layout::ColumnMajor,
                                                              0,
                                                              typename MmaCore::IteratorThreadMapB,
                                                              kAlignmentB>;

  using ThreadBlockMma =
      cutlass::gemm::threadblock::MmaSingleStageGemv<typename MmaCore::Shape,
                                                     IteratorA,
                                                     IteratorB,
                                                     ElementAccumulator,
                                                     cutlass::layout::RowMajor,
                                                     typename MmaCore::MmaPolicy>;
};


template<
  /// Element type for A matrix operand
  typename ElementA_,
  /// Access granularity of A matrix in units of elements
  int AlignmentA,
  /// Element type for B matrix operand
  typename ElementB_,
  /// Access granularity of B matrix in units of elements
  int AlignmentB,
  /// Element type for internal accumulation
  typename ElementAccumulator_,
  /// Tag indicating architecture to tune for
  typename ArchTag_,
  /// Threadblock-level tile size (concept: GemmShape)
  typename ThreadblockShape_,
  /// Warp-level tile size (concept: GemmShape)
  typename WarpShape_,
  /// Warp thread layout (concept: MatrixShape)
  typename WarpThreadArrangement_
>
struct DefaultMmaGemv<
 ElementA_,
 cutlass::layout::ColumnMajor,
 AlignmentA,
 ElementB_,
 cutlass::layout::RowMajor,
 AlignmentB,
 ElementAccumulator_,
 ArchTag_,
 ThreadblockShape_,
 WarpShape_,
 GemmShape<1, 1, 1>,
 WarpThreadArrangement_,
 arch::OpClassSimt
>{

  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using ElementAccumulator = ElementAccumulator_;
  using ArchTag = ArchTag_;

  using ThreadBlockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = GemmShape<1, 1, 1>;
  using WarpThreadArrangement = WarpThreadArrangement_;

  static const int kAlignmentA = AlignmentA;
  static const int kAlignmentB = AlignmentB;

  using MmaCore =
      typename cutlass::gemm::threadblock::DefaultMmaCoreGemv<ThreadBlockShape,
                                                              WarpShape,
                                                              InstructionShape,
                                                              WarpThreadArrangement,
                                                              ElementA,
                                                              cutlass::layout::ColumnMajor,
                                                              ElementB,
                                                              cutlass::layout::RowMajor,
                                                              ElementAccumulator,
                                                              cutlass::layout::ColumnMajor,
                                                              arch::OpClassSimt>;

  using IteratorA =
      cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
                                                              ElementA,
                                                              cutlass::layout::ColumnMajor,
                                                              1,
                                                              typename MmaCore::IteratorThreadMapA,
                                                              kAlignmentA>;

  static_assert(kAlignmentB == 1, "AlignmentB must be 1, when B matrix is RowMajor.");

  using IteratorB =
      cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kM>,
                                                              ElementB,
                                                              cutlass::layout::RowMajor,
                                                              0,
                                                              typename MmaCore::IteratorThreadMapB,
                                                              kAlignmentB>;

  using ThreadBlockMma =
      cutlass::gemm::threadblock::MmaSingleStageGemv<typename MmaCore::Shape,
                                                     IteratorA,
                                                     IteratorB,
                                                     ElementAccumulator,
                                                     cutlass::layout::ColumnMajor,
                                                     typename MmaCore::MmaPolicy>;
};

}
}
}