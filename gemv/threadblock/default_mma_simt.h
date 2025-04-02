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
  /// Access granularity of A matrix in units of elements
  int AlignmentA,
  /// Access granularity of B matrix in units of elements
  int AlignmentB,
  /// Threadblock-level tile size (concept: GemmShape)
  typename ThreadblockShape_,
  /// Warp-level tile size (concept: GemmShape)
  typename WarpShape_,
  /// Warp thread layout (concept: MatrixShape)
  typename WarpThreadArrangement_
>
struct DefaultMmaGemv<
 float,
 cutlass::layout::RowMajor,
 AlignmentA,
 float,
 cutlass::layout::ColumnMajor,
 AlignmentB,
 float,
 arch::Sm50,
 ThreadblockShape_,
 WarpShape_,
 GemmShape<1, 1, 1>,
 WarpThreadArrangement_,
 arch::OpClassSimt
>{

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
                                                              float,
                                                              cutlass::layout::RowMajor,
                                                              float,
                                                              cutlass::layout::ColumnMajor,
                                                              float,
                                                              cutlass::layout::RowMajor,
                                                              arch::OpClassSimt>;

  using IteratorA =
      cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
                                                              float,
                                                              cutlass::layout::RowMajor,
                                                              1,
                                                              typename MmaCore::IteratorThreadMapA,
                                                              kAlignmentA>;

  using IteratorB =
      cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kM>,
                                                              float,
                                                              cutlass::layout::ColumnMajor,
                                                              0,
                                                              typename MmaCore::IteratorThreadMapB,
                                                              kAlignmentB>;

  using ThreadBlockMma =
      cutlass::gemm::threadblock::MmaSingleStageGemv<typename MmaCore::Shape,
                                                     IteratorA,
                                                     IteratorB,
                                                     float,
                                                     cutlass::layout::RowMajor,
                                                     typename MmaCore::MmaPolicy>;
};

}
}
}