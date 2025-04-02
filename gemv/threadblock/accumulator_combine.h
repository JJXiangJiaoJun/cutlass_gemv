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

template <
 typename ElementC,
 typename LayoutC,
 int kElementCount,
 typename WarpThreadArrangement,
 typename WarpCount,
 int kThreadsPerGroup,
 int kWarpPartitionsK
>
class AccumulatorCombine;


///< Partial Specialization for thread combine
template <
  typename LayoutC_,
  int ElementCount,
  typename WarpThreadArrangement,
  typename WarpCount
>
class AccumulatorCombine<
  float,
  LayoutC_,
  ElementCount,
  WarpThreadArrangement,
  WarpCount,
  1,
  1
>{
public:
  using ElementC = float;
  using LayoutC = LayoutC_;
  static int const kElementCount = ElementCount;
  static int const kThreadsPerGroup = 1;
  static int const kWarpPartitionsK = 1;

  using FragmentC = cutlass::Array<ElementC, kElementCount>;

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
 typename LayoutC_,
 int ElementCount,
 typename WarpThreadArrangement,
 typename WarpCount,
 int ThreadsPerGroup
>
class AccumulatorCombine<
 float,
 LayoutC_,
 ElementCount,
 WarpThreadArrangement,
 WarpCount,
 ThreadsPerGroup,
 1
>{
public:
  using ElementC = float;
  using LayoutC = LayoutC_;
  static int const kElementCount = ElementCount;
  static int const kThreadsPerGroup = ThreadsPerGroup;
  static int const kWarpPartitionsK = 1;

  using FragmentC = cutlass::Array<ElementC, kElementCount>;
  using CombineOp =
      cutlass::gemm::warp::Reduce<ElementC, kElementCount, kThreadsPerGroup>;

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

///< Partial Specialization for threadblock combine
template <
 typename LayoutC_,
 int ElementCount,
 typename WarpThreadArrangement_,
 typename WarpCount_,
 int ThreadsPerGroup,
 int WarpPartitionsK
>
class AccumulatorCombine<
 float,
 LayoutC_,
 ElementCount,
 WarpThreadArrangement_,
 WarpCount_,
 ThreadsPerGroup,
 WarpPartitionsK
> {
public:
  using ElementC = float;
  using LayoutC = LayoutC_;
  using WarpThreadArrangement = WarpThreadArrangement_;
  using WarpCount = WarpCount_;
  static int const kElementCount = ElementCount;
  static int const kThreadsPerGroup = ThreadsPerGroup;
  static int const kWarpPartitionsK = WarpPartitionsK;

  using FragmentC = cutlass::Array<ElementC, kElementCount>;
  using CombineOp =
      cutlass::gemm::warp::Reduce<ElementC, kElementCount, kThreadsPerGroup>;

  using FragmentSharedReduce = cutlass::Array<ElementC, kWarpPartitionsK>;

public:
  struct SharedStorage {
  public:
    using AccumulatorShape =
        cutlass::MatrixShape<kElementCount * WarpThreadArrangement::kRow *
                                 WarpCount::kM,
                             kWarpPartitionsK>;

  public:
    cutlass::AlignedArray<ElementC, AccumulatorShape::kCount> acc_buffer;
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

        ElementC shared_acc = ElementC(0);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kWarpPartitionsK; i++) {
          shared_acc += frag_shared_reduce[i];
        }

        accumulator[r] = shared_acc;
      }
    }
  }
};

} // namespace threadblock
} // namespace gemm
} // namespace cutlass