#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"

#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/gemm_coord.h"

#include "cutlass/arch/mma.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

template<
 typename Mma_,
 typename AccumulatorCombine_,
 typename Epilogue_
>
struct Gemv {
public:

  using Mma = Mma_;
  using AccumulatorCombine = AccumulatorCombine_;
  using Epilogue = Epilogue_;
  using OutputOp = typename Epilogue::OutputOp;


  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreads = 32 * WarpCount::kCount;

  struct Params {
    cutlass::gemm::GemmCoord problem_size;

    typename Mma::IteratorA::Params params_A;
    typename Mma::IteratorA::TensorRef ref_A;
    typename Mma::IteratorB::Params params_B;
    typename Mma::IteratorB::TensorRef ref_B;
    typename Epilogue::OutputTileIterator::Params params_C;
    typename Epilogue::OutputTileIterator::TensorRef ref_C;
    typename Epilogue::OutputTileIterator::Params params_D;
    typename Epilogue::OutputTileIterator::TensorRef ref_D;
    typename OutputOp::Params output_op;


    CUTLASS_HOST_DEVICE
    Params() {}

    CUTLASS_HOST_DEVICE
    Params(cutlass::gemm::GemmCoord const &problem_size_,
           typename Mma::IteratorA::TensorRef ref_A_,
           typename Mma::IteratorB::TensorRef ref_B_,
           typename Epilogue::OutputTileIterator::TensorRef ref_C_,
           typename Epilogue::OutputTileIterator::TensorRef ref_D_,
           typename OutputOp::Params output_op_ = typename OutputOp::Params())
        : problem_size(problem_size_),
          params_A(ref_A_.layout()),
          ref_A(ref_A_),
          params_B(ref_B_.layout()),
          ref_B(ref_B_),
          params_C(ref_C_.layout()),
          ref_C(ref_C_),
          params_D(ref_D_.layout()),
          ref_D(ref_D_),
          output_op(output_op_)

    {}
  };

  using SharedStorage = typename AccumulatorCombine::SharedStorage;

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  Gemv() {}

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {
    cutlass::gemm::GemmCoord threadblock_tile_offset = cutlass::gemm::GemmCoord(blockIdx.x, 0, 0);

    if (threadblock_tile_offset.m() * Mma::Shape::kM >= params.problem_size.m()) {
      return;
    }

    // Compute initial location in logical coordinates
    cutlass::MatrixCoord tb_offset_A{
      threadblock_tile_offset.m() * Mma::Shape::kM,
      0
    };

    cutlass::MatrixCoord tb_offset_B{
      0,
      threadblock_tile_offset.n() * Mma::Shape::kN
    };

    int gemm_k_iterations = (params.problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

    int thread_idx = threadIdx.x;

    typename Mma::IteratorA iterator_A(
      params.params_A,
      params.ref_A.data(),
      {params.problem_size.m(), params.problem_size.k()},
      thread_idx,
      tb_offset_A
    );

    typename Mma::IteratorB iterator_B(
      params.params_B,
      params.ref_B.data(),
      {params.problem_size.k(), params.problem_size.m()},
      thread_idx,
      tb_offset_B
    );

    Mma mma;

    typename Mma::FragmentC accumulators;

    accumulators.clear();

    mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);

    // if (blockIdx.x == 0 && warp_idx == 0 && lane_idx == 4) {
    //   for (int i = 0; i < accumulators.size(); i++) {
    //     printf("block %d, warp %d, lane %d, mma[%d]=%f\n",
    //            blockIdx.x,
    //            warp_idx,
    //            lane_idx,
    //            i,
    //            accumulators[i]);
    //   }
    // }

    ///< accumulator combine
    AccumulatorCombine acc_combine;
    acc_combine(shared_storage, accumulators);

    // if (blockIdx.x == 0 && warp_idx == 0 && lane_idx == 4) {
    //   for (int i = 0; i < accumulators.size(); i++) {
    //     printf("block %d, warp %d, lane %d, acc[%d]=%f\n",
    //            blockIdx.x,
    //            warp_idx,
    //            lane_idx,
    //            i,
    //            accumulators[i]);
    //   }
    // }

    cutlass::MatrixCoord tb_offset_C{
      threadblock_tile_offset.m() * Mma::Shape::kM,
      0
    };

    cutlass::MatrixCoord tb_offset_D{
      threadblock_tile_offset.m() * Mma::Shape::kM,
      0
    };

    typename Epilogue::OutputTileIterator Iterator_C(
      params.params_C,
      params.ref_C.data(),
      {params.problem_size.m(), params.problem_size.n()},
      thread_idx,
      tb_offset_C
    );

    typename Epilogue::OutputTileIterator Iterator_D(
      params.params_D,
      params.ref_D.data(),
      {params.problem_size.m(), params.problem_size.n()},
      thread_idx,
      tb_offset_D
    );

    OutputOp output_op(params.output_op);
    Epilogue epilogue;
    epilogue(output_op, Iterator_D, accumulators, Iterator_C);
  }
};

}
}
}