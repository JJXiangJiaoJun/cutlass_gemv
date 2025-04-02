#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/fast_math.h"

#include "cutlass/numeric_types.h"
#include "cutlass/pitch_linear_coord.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/layout/matrix.h"


#include "gemv/transform/pitch_linear_thread_map.h"
#include "gemv/warp/mma_simt_policy.h"
#include "gemv/warp/mma_simt.h"
#include "gemv/threadblock/default_mma_core.h"

namespace cutlass {
namespace gemm {
namespace threadblock {

template<
    /// Shape of threadblock-scoped matrix multiply operator
    typename Shape_,
    /// Shape of warp-level matrix multiply operator
    typename WarpShape_,
    /// Shape of warp thread layout (concept: MatrixShape)
    typename WarpThreadArrangement_,
    /// Element data type of A operand
    typename ElementA_,
    /// Element data type of B operand
    typename ElementB_,
    /// Data type of accumulator
    typename ElementC_
>
struct DefaultMmaCoreGemv<
  Shape_,
  WarpShape_,
  GemmShape<1, 1, 1>,
  WarpThreadArrangement_,
  ElementA_,
  cutlass::layout::RowMajor,
  ElementB_,
  cutlass::layout::ColumnMajor,
  ElementC_,
  cutlass::layout::RowMajor,
  arch::OpClassSimt
>{

  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = GemmShape<1, 1, 1>;
  using WarpThreadArrangement = WarpThreadArrangement_;
  using ElementA = ElementA_;
  using LayoutA  = cutlass::layout::RowMajor;
  using ElementB = ElementB_;
  using LayoutB  = cutlass::layout::ColumnMajor;
  using ElementC = ElementC_;
  using LayoutC  = cutlass::layout::RowMajor;
  using OperatorClass = arch::OpClassSimt;

  using WarpCount = GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN,
    Shape::kK / WarpShape::kK
  >;

  static_assert(Shape::kN == 1, "");
  static_assert(WarpShape::kN == 1, "");
  static_assert(WarpCount::kN == 1, "");


  /// Number of threads per warp
  static int const kWarpSize = 32;

  static_assert(WarpThreadArrangement::kCount == kWarpSize, "");

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  static int const kAccessSizeInBits = 128;

  static int const kElementsPerAccess = kAccessSizeInBits / sizeof_bits<ElementA>::value;

  using UnderlyingWarpThreadArrangement =
      cutlass::PitchLinearShape<WarpThreadArrangement::kColumn, WarpThreadArrangement::kRow>;

  using IteratorThreadMapA = cutlass::transform::PitchLinearWarpStripminedThreadMap<
    cutlass::PitchLinearShape<Shape::kK, Shape::kM>,
    kThreads,
    cutlass::PitchLinearShape<WarpCount::kK, WarpCount::kM>,
    UnderlyingWarpThreadArrangement,
    kElementsPerAccess
  >;

  using IteratorThreadMapB = cutlass::transform::PitchLinearWarpStripminedThreadMap<
    cutlass::PitchLinearShape<Shape::kK, Shape::kM>,
    kThreads,
    cutlass::PitchLinearShape<WarpCount::kK, WarpCount::kM>,
    UnderlyingWarpThreadArrangement,
    kElementsPerAccess
  >;

  using Policy = cutlass::gemm::warp::MmaSimtGemvPolicy<
      cutlass::MatrixShape<UnderlyingWarpThreadArrangement::kStrided,
                           UnderlyingWarpThreadArrangement::kContiguous>,
      InstructionShape>;

  using RegisterTileShape =
      cutlass::MatrixShape<IteratorThreadMapA::Iterations::kStrided,
                           IteratorThreadMapA::Iterations::kContiguous * kElementsPerAccess>;

  using MmaWarpSimtGemv = cutlass::gemm::warp::MmaSimtGemv<WarpShape,
                                                           RegisterTileShape,
                                                           ElementA,
                                                           LayoutA,
                                                           ElementB,
                                                           LayoutB,
                                                           ElementC,
                                                           LayoutC,
                                                           Policy>;

  using MmaPolicy = cutlass::gemm::warp::MmaGemvPolicy<MmaWarpSimtGemv, WarpCount::kK>;
};


}
}
}