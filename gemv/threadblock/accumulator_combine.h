#pragma once

#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"

#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"

#include "cutlass/arch/mma.h"

#include "gemv/warp/reduce.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

/// threadblock combine
template <
 typename ElementAccumulator_,
 typename LayoutAccumulator_,
 int ElementCount,
 typename WarpThreadArrangement_,
 typename WarpCount_,
 int ThreadsPerGroup,
 int WarpPartitionsK
>
class AccumulatorCombine {
public:
  using ElementAccumulator = ElementAccumulator_;
  using LayoutAccumulator = LayoutAccumulator_;
  using WarpThreadArrangement = WarpThreadArrangement_;
  using WarpCount = WarpCount_;

  static int const kElementCount = ElementCount;
  static int const kThreadsPerGroup = ThreadsPerGroup;
  static int const kWarpPartitionsK = WarpPartitionsK;

  using FragmentC = cutlass::Array<ElementAccumulator, kElementCount>;
  using CombineOp =
      cutlass::gemm::warp::Reduce<ElementAccumulator, kElementCount, kThreadsPerGroup>;

  using FragmentSharedReduce = cutlass::Array<ElementAccumulator, kWarpPartitionsK>;

public:
  struct SharedStorage {
  public:
    using AccumulatorShape =
        cutlass::MatrixShape<kElementCount * WarpThreadArrangement::kRow *
                                 WarpCount::kM,
                             kWarpPartitionsK>;

  public:
    cutlass::AlignedArray<ElementAccumulator, AccumulatorShape::kCount> acc_buffer;
  };

public:
  CUTLASS_HOST_DEVICE
  AccumulatorCombine() {}

  CUTLASS_DEVICE
  void operator()(SharedStorage &shared_storage, FragmentC &accumulator) {

    int thread_idx = threadIdx.x;
    int warp_idx = thread_idx / 32;
    int lane_idx = thread_idx % 32;

    int warp_row_idx = warp_idx / WarpCount::kK;
    int warp_col_idx = warp_idx % WarpCount::kK;

    int lane_row_idx = lane_idx / WarpThreadArrangement::kColumn;
    int lane_col_idx = lane_idx % WarpThreadArrangement::kColumn;

    CombineOp combine_op;
    ///< Step 1. warp reduce
    combine_op(accumulator);

    ///< Step 2. write back to shared memory
    int warp_row_offset =
        warp_row_idx * kElementCount * WarpThreadArrangement::kRow;
    int lane_row_offset = lane_row_idx;
    int lane_col_offset = warp_col_idx;

    if (lane_col_idx == 0) {
      CUTLASS_PRAGMA_UNROLL
      for (int r = 0; r < kElementCount; r++) {
        int shared_offset = (warp_row_offset + r * WarpThreadArrangement::kRow +
                             lane_row_offset) *
                                kWarpPartitionsK +
                            lane_col_offset;
        *(shared_storage.acc_buffer.data() + shared_offset) = accumulator[r];
      }
    }

    __syncthreads();

    ///< Step 3. Shared memory reduce
    if (warp_col_idx == 0 && lane_col_idx == 0) {

      CUTLASS_PRAGMA_UNROLL
      for (int r = 0; r < kElementCount; r++) {

        int shared_offset = (warp_row_offset + r * WarpThreadArrangement::kRow +
                             lane_row_offset) *
                            kWarpPartitionsK;

        FragmentSharedReduce frag_shared_reduce;
        cutlass::arch::shared_load<FragmentSharedReduce>(
            frag_shared_reduce,
            shared_storage.acc_buffer.data() + shared_offset);

        ElementAccumulator shared_acc = ElementAccumulator(0);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kWarpPartitionsK; i++) {
          shared_acc += *(frag_shared_reduce.data() + i);
        }

        *(accumulator.data() + r) = shared_acc;
      }
    }
  }
};

///< Partial Specialization for thread combine
template <
  typename ElementAccumulator_,
  typename LayoutAccumulator_,
  int ElementCount,
  typename WarpThreadArrangement,
  typename WarpCount
>
class AccumulatorCombine<
  ElementAccumulator_,
  LayoutAccumulator_,
  ElementCount,
  WarpThreadArrangement,
  WarpCount,
  1,
  1
>{
public:
  using ElementAccumulator = ElementAccumulator_;
  using LayoutAccumulator = LayoutAccumulator_;
  static int const kElementCount = ElementCount;
  static int const kThreadsPerGroup = 1;
  static int const kWarpPartitionsK = 1;

  using FragmentC = cutlass::Array<ElementAccumulator, kElementCount>;

public:
  struct SharedStorage {};

public:
  CUTLASS_HOST_DEVICE
  AccumulatorCombine() {}

  CUTLASS_DEVICE
  void operator()(SharedStorage &shared_storage, FragmentC &accumulator) {}
};


///< Partial Specialization for warp combine
template <
 typename ElementAccumulator_,
 typename LayoutAccumulator_,
 int ElementCount,
 typename WarpThreadArrangement,
 typename WarpCount,
 int ThreadsPerGroup
>
class AccumulatorCombine<
 ElementAccumulator_,
 LayoutAccumulator_,
 ElementCount,
 WarpThreadArrangement,
 WarpCount,
 ThreadsPerGroup,
 1
>{
public:
  using ElementAccumulator = ElementAccumulator_;
  using LayoutAccumulator = LayoutAccumulator_;
  static int const kElementCount = ElementCount;
  static int const kThreadsPerGroup = ThreadsPerGroup;
  static int const kWarpPartitionsK = 1;

  using FragmentC = cutlass::Array<ElementAccumulator, kElementCount>;
  using CombineOp =
      cutlass::gemm::warp::Reduce<ElementAccumulator, kElementCount, kThreadsPerGroup>;

public:
  struct SharedStorage {};

public:
  CUTLASS_HOST_DEVICE
  AccumulatorCombine() {}

  CUTLASS_DEVICE
  void operator()(SharedStorage &shared_storage, FragmentC &accumulator) {
    CombineOp combine_op;
    combine_op(accumulator);
  }
};


} // namespace threadblock
} // namespace gemm
} // namespace cutlass