#pragma once


#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/vector.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/aligned_buffer.h"

#include "cutlass/pitch_linear_coord.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

#include "epilogue/threadblock/output_tile_thread_map.h"
#include "epilogue/threadblock/epilogue.h"


namespace cutlass {
namespace epilogue {
namespace threadblock {

template <
 typename Shape,
 typename OutputOp,
 typename MmaCore,
 typename LayoutC,
 int ElementPerAccess
>
struct DefaultEpilogueGemvSimt;

template <
 typename Shape_,
 typename OutputOp_,
 typename MmaCore_,
 int ElementPerAccess
>
struct DefaultEpilogueGemvSimt<
 Shape_,
 OutputOp_,
 MmaCore_,
 cutlass::layout::RowMajor,
 ElementPerAccess
> {
  using Shape = Shape_;
  using OutputOp = OutputOp_;
  using MmaCore = MmaCore_;
  using MmaPolicy = typename MmaCore::MmaPolicy;
  using ElementOutput = typename OutputOp::ElementOutput;
  using ElementAccumulator = typename MmaPolicy::Operator::ElementC;
  using LayoutC = cutlass::layout::RowMajor;

  static const int kElementsPerAccess = ElementPerAccess;

  //
  // Thread map
  //
  using OutputTileThreadMap = cutlass::epilogue::threadblock::PitchLinearWarpVectorThreadMap<
      cutlass::PitchLinearShape<Shape::kN, Shape::kM>,
      MmaCore::kThreads,
      cutlass::PitchLinearShape<MmaCore::WarpCount::kK, MmaCore::WarpCount::kM>,
      typename MmaCore::UnderlyingWarpThreadArrangement,
      kElementsPerAccess>;


  using OutputTileIterator = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<Shape::kM, Shape::kN>,
      ElementOutput,
      cutlass::layout::RowMajor,
      1,
      OutputTileThreadMap>;

  using Epilogue = cutlass::epilogue::threadblock::
      EpilogueGemv<Shape, ElementAccumulator, OutputTileIterator, OutputOp>;
};

template <
 typename Shape_,
 typename OutputOp_,
 typename MmaCore_,
 int ElementPerAccess
>
struct DefaultEpilogueGemvSimt<
 Shape_,
 OutputOp_,
 MmaCore_,
 cutlass::layout::ColumnMajor,
 ElementPerAccess
> {
  using Shape = Shape_;
  using OutputOp = OutputOp_;
  using MmaCore = MmaCore_;
  using MmaPolicy = typename MmaCore::MmaPolicy;
  using ElementOutput = typename OutputOp::ElementOutput;
  using ElementAccumulator = typename MmaPolicy::Operator::ElementC;
  using LayoutC = cutlass::layout::ColumnMajor;

  static const int kElementsPerAccess = ElementPerAccess;

  //
  // Thread map
  //
  using OutputTileThreadMap = cutlass::epilogue::threadblock::PitchLinearWarpVectorThreadMap<
      cutlass::PitchLinearShape<Shape::kM, Shape::kN>,
      MmaCore::kThreads,
      cutlass::PitchLinearShape<MmaCore::WarpCount::kM, MmaCore::WarpCount::kK>,
      typename MmaCore::UnderlyingWarpThreadArrangement,
      kElementsPerAccess>;


  using OutputTileIterator = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<Shape::kM, Shape::kN>,
      ElementOutput,
      cutlass::layout::ColumnMajor,
      1,
      OutputTileThreadMap>;

  using Epilogue = cutlass::epilogue::threadblock::
      EpilogueGemv<Shape, ElementAccumulator, OutputTileIterator, OutputOp>;
};


} /// end of namespace threadblock
} /// end of namespace epilogue
} /// end of namespace cutlass