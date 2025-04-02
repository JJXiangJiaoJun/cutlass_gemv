#pragma once

#include "cutlass/array.h"
#include "cutlass/coord.h"
#include "cutlass/predicate_vector.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/tensor_view.h"
#include "cutlass/layout/pitch_linear.h"


namespace cutlass {
namespace transform {

template <
  typename Shape_,
  int Threads,
  typename WarpArrangement_,
  typename WarpThreadArrangement_,
  int ElementsPerAccess = 1
>
struct PitchLinearWarpStripminedThreadMap {

  /// Tensor coordinate
  using TensorCoord = layout::PitchLinearCoord;

  /// Tile shape
  using Shape = Shape_;

  /// Number of threads total
  static int const kThreads = Threads;

  /// Extract vector length from Layout
  static int const kElementsPerAccess = ElementsPerAccess;

  /// Shape of access by each thread
  using ThreadAccessShape = layout::PitchLinearShape<kElementsPerAccess, 1>;

  /// Internal details made public to facilitate introspection
  struct Detail {

    /// Fixed arrangement of threads within a warp (units of threads).
    using WarpThreadArrangement = WarpThreadArrangement_;

    /// Number of threads per warp
    static int const kWarpSize = WarpThreadArrangement::kCount;

    /// Number of participating warps
    static int const kWarpCount = kThreads / kWarpSize;

    static_assert(
      !(Shape::kContiguous % kElementsPerAccess),
      "Shape must be divisible by vector length.");

    /// Compute the 'shape' of the overall tile in units of vectors
    using ShapeInAccesses = layout::PitchLinearShape<
      Shape::kContiguous / kElementsPerAccess,
      Shape::kStrided
    >;

    static_assert(
      !(ShapeInAccesses::kContiguous % WarpThreadArrangement::kContiguous),
      "ShapeInAccesses must be divisible by WarpThreadArrangement.");

    static_assert(
      !(ShapeInAccesses::kStrided % WarpThreadArrangement::kStrided),
      "ShapeInAccesses must be divisible by WarpThreadArrangement.");

    // compute number of warp-level accesses total
    using WarpAccessIterations = layout::PitchLinearShape<
      ShapeInAccesses::kContiguous / WarpThreadArrangement::kContiguous,
      ShapeInAccesses::kStrided / WarpThreadArrangement::kStrided
    >;

    using WarpArrangement = WarpArrangement_;

    // Divide it into the number of warps, first partitioning the strided dimension then the
    // contiguous.
    static int const kWarpsStrided = WarpArrangement::kStrided;

    static int const kWarpsContiguous = WarpArrangement::kContiguous;
    /// Arrangement of warps within a threadblock-scoped tile
  };

  ///< Iterations along each dimension (concept: PitchLinearShape)
  using Iterations = layout::PitchLinearShape<
    Detail::WarpAccessIterations::kContiguous / Detail::kWarpsContiguous,
    Detail::WarpAccessIterations::kStrided / Detail::kWarpsStrided
  >;

  static_assert(Iterations::kCount,
    "Number of iterations must be non-zero");

  ///< Delta betweeen accesses (units of elements, concept: PitchLinearShape)
  using Delta = layout::PitchLinearShape<
    Detail::WarpThreadArrangement::kContiguous * kElementsPerAccess,
    Detail::WarpThreadArrangement::kStrided
  >;

  /// Maps thread ID to a coordinate offset within the tensor's logical coordinate space
  CUTLASS_HOST_DEVICE
  static TensorCoord initial_offset(int thread_id) {

    int warp_id = (thread_id / Detail::kWarpSize);
    int lane_id = (thread_id % Detail::kWarpSize);

    //
    // compute warp-level offset
    //

    // This is the shape of the entire area covered by a warp's memory access (in units of vectors)
    layout::PitchLinearCoord warp_footprint{
      Detail::WarpThreadArrangement::kContiguous * Iterations::kContiguous,
      Detail::WarpThreadArrangement::kStrided * Iterations::kStrided
    };

    // This is the offset of a specific warp (in units of vectors)
    layout::PitchLinearCoord warp_offset{
      (warp_id % Detail::kWarpsContiguous),
      (warp_id / Detail::kWarpsContiguous)
    };

    // This is the offset of a specific thread within a warp (units of vectors)
    layout::PitchLinearCoord thread_offset_in_warp{
      lane_id % Detail::WarpThreadArrangement::kContiguous,
      lane_id / Detail::WarpThreadArrangement::kContiguous
    };

    // This is the offset of a thread within a threadblock tile (units of vectors)
    layout::PitchLinearCoord thread_offset_in_threadblock_tile_vec =
      warp_footprint * warp_offset + thread_offset_in_warp;

    // This is the offset of a thread within a threadblock tile (units of elements)
    layout::PitchLinearCoord thread_offset_in_threadblock_tile_base{
      thread_offset_in_threadblock_tile_vec.contiguous() * kElementsPerAccess,
      thread_offset_in_threadblock_tile_vec.strided()
    };

    return thread_offset_in_threadblock_tile_base;
  }
};
}  /// end of namespace transform
}  /// end of namespace cutlass
