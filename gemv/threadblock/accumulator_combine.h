#pragma once

#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"

#include "cutlass/arch/mma.h"

#include "gemv/warp/reduce.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

template <
 typename ElementAccumulator,
 typename LayoutAccumulator,
 int ElementCount,
 int ElementPerAccess,
 typename WarpThreadArrangement,
 typename WarpCount,
 int ThreadsPerGroup,
 int WarpPartitionsK
>
class AccumulatorCombine;

/// ====================================================================================
///< Partial Specialization for thread combine, RowMajor
template <
  typename ElementAccumulator_,
  int ElementCount,
  int ElementPerAccess,
  typename WarpThreadArrangement,
  typename WarpCount
>
class AccumulatorCombine<
  ElementAccumulator_,
  cutlass::layout::RowMajor,
  ElementCount,
  ElementPerAccess,
  WarpThreadArrangement,
  WarpCount,
  1,
  1
>{
public:
  using ElementAccumulator = ElementAccumulator_;
  using LayoutAccumulator = cutlass::layout::RowMajor;

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

/// ====================================================================================
///< Partial Specialization for thread combine, ColumnMajor
template <
  typename ElementAccumulator_,
  int ElementCount,
  int ElementPerAccess,
  typename WarpThreadArrangement,
  typename WarpCount
>
class AccumulatorCombine<
  ElementAccumulator_,
  cutlass::layout::ColumnMajor,
  ElementCount,
  ElementPerAccess,
  WarpThreadArrangement,
  WarpCount,
  1,
  1
>{
public:
  using ElementAccumulator = ElementAccumulator_;
  using LayoutAccumulator = cutlass::layout::ColumnMajor;

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

/// ====================================================================================
///< Partial Specialization for warp combine, RowMajor
template <
 typename ElementAccumulator_,
 int ElementCount,
 int ElementPerAccess,
 typename WarpThreadArrangement,
 typename WarpCount,
 int ThreadsPerGroup
>
class AccumulatorCombine<
 ElementAccumulator_,
 cutlass::layout::RowMajor,
 ElementCount,
 ElementPerAccess,
 WarpThreadArrangement,
 WarpCount,
 ThreadsPerGroup,
 1
>{
public:
  using ElementAccumulator = ElementAccumulator_;
  using LayoutAccumulator = cutlass::layout::RowMajor;

  static int const kElementCount = ElementCount;
  static int const kThreadsPerGroup = ThreadsPerGroup;
  static int const kWarpPartitionsK = 1;

  using FragmentC = cutlass::Array<ElementAccumulator, kElementCount>;
  using CombineOp =
      cutlass::gemm::warp::Reduce<ElementAccumulator, kElementCount, kThreadsPerGroup, LayoutAccumulator>;

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

/// ====================================================================================
///< Partial Specialization for warp combine, ColumnMajor
template <
 typename ElementAccumulator_,
 int ElementCount,
 int ElementPerAccess,
 typename WarpThreadArrangement,
 typename WarpCount,
 int ThreadsPerGroup
>
class AccumulatorCombine<
 ElementAccumulator_,
 cutlass::layout::ColumnMajor,
 ElementCount,
 ElementPerAccess,
 WarpThreadArrangement,
 WarpCount,
 ThreadsPerGroup,
 1
>{
public:
  using ElementAccumulator = ElementAccumulator_;
  using LayoutAccumulator = cutlass::layout::ColumnMajor;

  static int const kElementCount = ElementCount;
  static int const kThreadsPerGroup = ThreadsPerGroup;
  static int const kWarpPartitionsK = 1;

  using FragmentC = cutlass::Array<ElementAccumulator, kElementCount>;
  using CombineOp =
      cutlass::gemm::warp::Reduce<ElementAccumulator, kElementCount, kThreadsPerGroup, LayoutAccumulator>;

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

/// ====================================================================================
///< Partial Specialization for threadblock combine, RowMajor
template <
 typename ElementAccumulator_,
 int ElementCount,
 typename WarpThreadArrangement_,
 typename WarpCount_,
 int ThreadsPerGroup,
 int WarpPartitionsK
>
class AccumulatorCombine<
 ElementAccumulator_,
 cutlass::layout::RowMajor,
 ElementCount,
 1,
 WarpThreadArrangement_,
 WarpCount_,
 ThreadsPerGroup,
 WarpPartitionsK
> {
public:
  using ElementAccumulator = ElementAccumulator_;
  using LayoutAccumulator = cutlass::layout::RowMajor;
  using WarpThreadArrangement = WarpThreadArrangement_;
  using WarpCount = WarpCount_;

  static int const kElementCount = ElementCount;
  static int const kThreadsPerGroup = ThreadsPerGroup;
  static int const kWarpPartitionsK = WarpPartitionsK;

  using FragmentC = cutlass::Array<ElementAccumulator, kElementCount>;
  using CombineOp =
      cutlass::gemm::warp::Reduce<ElementAccumulator, kElementCount, kThreadsPerGroup, LayoutAccumulator>;

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
        *(shared_storage.acc_buffer.data() + shared_offset) = *(accumulator.data() + r);
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


/// ====================================================================================
///< Partial Specialization for threadblock combine, ColumnMajor
template <
 typename ElementAccumulator_,
 int ElementCount,
 int ElementPerAccess,
 typename WarpThreadArrangement_,
 typename WarpCount_,
 int ThreadsPerGroup,
 int WarpPartitionsK
>
class AccumulatorCombine<
 ElementAccumulator_,
 cutlass::layout::ColumnMajor,
 ElementCount,
 ElementPerAccess,
 WarpThreadArrangement_,
 WarpCount_,
 ThreadsPerGroup,
 WarpPartitionsK
> {
public:
  using ElementAccumulator = ElementAccumulator_;
  using LayoutAccumulator = cutlass::layout::ColumnMajor;
  using WarpThreadArrangement = WarpThreadArrangement_;
  using WarpCount = WarpCount_;

  static int const kElementPerAccess = ElementPerAccess;
  static int const kElementCount = ElementCount;
  static int const kIterations = kElementCount / kElementPerAccess;
  static int const kThreadsPerGroup = ThreadsPerGroup;
  static int const kWarpPartitionsK = WarpPartitionsK;

  static_assert(kElementCount % kElementPerAccess == 0, "ElementCount must be multiple of ElementPerAccess");

  using AccessType = cutlass::Array<ElementAccumulator, kElementPerAccess>;
  using FragmentC = cutlass::Array<ElementAccumulator, kElementCount>;

  using CombineOp =
      cutlass::gemm::warp::Reduce<ElementAccumulator, kElementCount, kThreadsPerGroup, LayoutAccumulator>;

  using FragmentSharedReduce = cutlass::Array<ElementAccumulator, kWarpPartitionsK>;

  using SharedAccumulatorOp = cutlass::plus<AccessType>;

public:
  struct SharedStorage {
  public:
    // using AccumulatorShape =
    //     cutlass::MatrixShape<kElementCount * WarpThreadArrangement::kRow * WarpCount::kM,
    //                          kWarpPartitionsK>;

    using AccumulatorShape =
        cutlass::MatrixShape<kWarpPartitionsK,
                             kElementCount * WarpThreadArrangement::kRow * WarpCount::kM>;

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

    int warp_row_idx = warp_idx % WarpCount::kM;
    int warp_col_idx = warp_idx / WarpCount::kM;

    int lane_row_idx = lane_idx % WarpThreadArrangement::kRow;
    int lane_col_idx = lane_idx / WarpThreadArrangement::kRow;

    CombineOp combine_op;

    ///< Step 1. warp reduce
    combine_op(accumulator);


    AccessType *access_ptr = reinterpret_cast<AccessType *>(&accumulator);

    ///< Step 2. write back to shared memory
    int warp_row_offset =
        warp_row_idx * kElementCount * WarpThreadArrangement::kRow;
    int lane_row_offset = lane_row_idx;
    int lane_col_offset = warp_col_idx;

    if (lane_col_idx == 0) {
      CUTLASS_PRAGMA_UNROLL
      for (int iter = 0; iter < kIterations; ++iter) {
        int shared_offset = (warp_row_offset + (iter * WarpThreadArrangement::kRow + lane_row_offset) * kElementPerAccess) +
                             lane_col_offset * SharedStorage::AccumulatorShape::kColumn;
        *reinterpret_cast<AccessType *>(shared_storage.acc_buffer.data() + shared_offset) = *(access_ptr + iter);
      }
    }

    __syncthreads();

    SharedAccumulatorOp shared_acc_op;

    ///< Step 3. Shared memory reduce
    if (warp_col_idx == 0 && lane_col_idx == 0) {

      CUTLASS_PRAGMA_UNROLL
      for (int iter = 0; iter < kIterations; ++iter) {

        AccessType frag_shared_reduce;
        frag_shared_reduce.clear();

        CUTLASS_PRAGMA_UNROLL
        for (int r = 0; r < kWarpPartitionsK; ++r) {

          int shared_offset = (warp_row_offset + (iter * WarpThreadArrangement::kRow + lane_row_offset) * kElementPerAccess) +
                               r * SharedStorage::AccumulatorShape::kColumn;

          AccessType frag_shared_load;
          cutlass::arch::shared_load<AccessType>(
              frag_shared_load,
              shared_storage.acc_buffer.data() + shared_offset);

          *reinterpret_cast<AccessType *>(&frag_shared_reduce) = shared_acc_op(frag_shared_reduce, frag_shared_load);
        }

        *reinterpret_cast<AccessType *>(access_ptr + iter) = frag_shared_reduce;
     }
   }
 }

};

} // namespace threadblock
} // namespace gemm
} // namespace cutlass