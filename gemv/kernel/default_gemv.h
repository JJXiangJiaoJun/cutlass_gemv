#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

#include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "gemv/kernel/gemv.h"
#include "gemv/threadblock/accumulator_combine.h"
#include "gemv/threadblock/default_mma.h"
#include "gemv/threadblock/default_mma_simt.h"
#include "epilogue/threadblock/default_epilogue_simt.h"

namespace cutlass {
namespace gemm {
namespace kernel {

template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Layout type for C and D matrix operands
    typename LayoutC,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Warp thread layout (concept: MatrixShape)
    typename WarpThreadArrangement,
    /// Epilogue output operator
    typename EpilogueOutputOp
>
struct DefaultGemv;

template<
  /// Element type for A matrix operand
  typename ElementA,
  int kAlignmentA,
  /// Element type for B matrix operand
  typename ElementB,
  int kAlignmentB,
  typename ElementC,
  /// Element type for internal accumulation
  typename ElementAccumulator,
  /// Tag indicating architecture to tune for
  typename ArchTag,
  /// Threadblock-level tile size (concept: GemmShape)
  typename ThreadBlockShape,
  /// Warp-level tile size (concept: GemmShape)
  typename WarpShape,
  /// Warp thread layout (concept: MatrixShape)
  typename WarpThreadArrangement,
  /// Epilogue output operator
  typename EpilogueOutputOp
>
struct DefaultGemv<
 ElementA,
 cutlass::layout::RowMajor,
 kAlignmentA,
 ElementB,
 cutlass::layout::ColumnMajor,
 kAlignmentB,
 ElementC,
 cutlass::layout::RowMajor,
 ElementAccumulator,
 arch::OpClassSimt,
 ArchTag,
 ThreadBlockShape,
 WarpShape,
 GemmShape<1, 1, 1>,
 WarpThreadArrangement,
 EpilogueOutputOp
> {
  using DefaultMma =
      typename cutlass::gemm::threadblock::DefaultMmaGemv<ElementA,
                                                          cutlass::layout::RowMajor,
                                                          kAlignmentA,
                                                          ElementB,
                                                          cutlass::layout::ColumnMajor,
                                                          kAlignmentB,
                                                          ElementAccumulator,
                                                          ArchTag,
                                                          ThreadBlockShape,
                                                          WarpShape,
                                                          GemmShape<1, 1, 1>,
                                                          WarpThreadArrangement,
                                                          arch::OpClassSimt>;
  using MmaCore = typename DefaultMma::MmaCore;
  using Mma = typename DefaultMma::ThreadBlockMma;

  using AccumulatorCombine = cutlass::gemm::threadblock::AccumulatorCombine<
      ElementAccumulator,
      cutlass::layout::RowMajor,
      Mma::FragmentC::kElements,
      cutlass::MatrixShape<MmaCore::UnderlyingWarpThreadArrangement::kStrided,
                           MmaCore::UnderlyingWarpThreadArrangement::kContiguous>,
      typename MmaCore::WarpCount,
      MmaCore::UnderlyingWarpThreadArrangement::kContiguous,
      MmaCore::MmaPolicy::kWarpPartitionsK>;


  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueGemvSimt<
      ThreadBlockShape,
      EpilogueOutputOp,
      MmaCore,
      EpilogueOutputOp::kCount>::Epilogue;

  using Kernel = cutlass::gemm::kernel::Gemv<Mma, AccumulatorCombine, Epilogue>;
};

}
}
}