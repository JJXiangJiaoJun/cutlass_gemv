#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/aligned_buffer.h"

#include "cutlass/numeric_types.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/matrix_shape.h"

namespace cutlass {
namespace gemm {
namespace threadblock {

template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
  /// Iterates over tiles of A operand in global memory
  //  (concept: ReadableTileIterator | ForwardTileIterator | MaskedTileIterator)
  typename IteratorA_,
  /// Iterates over tiles of B operand in global memory
  //  (concept: ReadableTileIterator | ForwardTileIterator | MaskedTileIterator)
  typename IteratorB_,
  /// Data type of accumulator matrix
  typename ElementC_,
  /// Data type of accumulator matrix
  typename LayoutC_,
  /// Policy describing tuning details (concept: MmaPolicy)
  typename Policy_
>
class MmaSingleStageGemv {
public:
  using Shape = Shape_;             ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using IteratorA = IteratorA_;     ///< Iterates over tiles of A operand in global memory
  using IteratorB = IteratorB_;     ///< Iterates over tiles of B operand in global memory
  using ElementC = ElementC_;       ///< Data type of accumulator matrix
  using LayoutC = LayoutC_;         ///< Layout of accumulator matrix
  using Policy = Policy_;           ///< Policy describing tuning details

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  using FragmentA = typename IteratorA::Fragment;
  using FragmentB = typename IteratorB::Fragment;
  /// Fragment of accumulator tile
  using FragmentC = typename Operator::FragmentC;


  using WarpTileIteratorA = typename Operator::IteratorA;
  using WarpTileIteratorB = typename Operator::IteratorB;

  using WarpTileFragmentA = typename Operator::FragmentA;
  using WarpTileFragmentB = typename Operator::FragmentB;


  /// Shape describing the overall GEMM computed from shared memory
  /// by each warp.
  using WarpGemm = typename Operator::Shape;

  /// Shape describing the number of warps filling the CTA
  using WarpCount = GemmShape<Shape::kM / WarpGemm::kM,
                              Shape::kN / WarpGemm::kN,
                              Shape::kK / WarpGemm::kK>;

  static const int kWarpGemmIterations =
      (WarpGemm::kK / Operator::Policy::WarpShape::kColumn / Operator::InstructionShape::kK);

  /// Number of stages
  static int const kStages = 1;

  struct SharedStorage {};


public:
  CUTLASS_HOST_DEVICE
  MmaSingleStageGemv() {}


  CUTLASS_DEVICE
  void operator()(
    int gemm_k_iterations,         ///< number of iterations of the mainloop
    FragmentC &accum,              ///< destination accumulator tile
    IteratorA iterator_A,          ///< iterator over A operand in global memory
    IteratorB iterator_B,          ///< iterator over B operand in global memory
    FragmentC const &src_accum) {  ///< source accumualtor tile

    accum = src_accum;

    FragmentA tb_frag_A;
    FragmentB tb_frag_B;

    tb_frag_A.clear();
    tb_frag_B.clear();

    iterator_A.load(tb_frag_A);
    iterator_B.load(tb_frag_B);

    ++iterator_A;
    ++iterator_B;

    WarpTileIteratorA warp_tile_iterator_A(tb_frag_A);
    WarpTileIteratorB warp_tile_iterator_B(tb_frag_B);

    WarpTileFragmentA warp_tile_frag_A;
    WarpTileFragmentB warp_tile_frag_B;

    Operator warp_mma;

    // Avoid reading out of bounds
    iterator_A.clear_mask(gemm_k_iterations <= 1);
    iterator_B.clear_mask(gemm_k_iterations <= 1);

    for(; gemm_k_iterations > 0; --gemm_k_iterations) {

      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < kWarpGemmIterations; ++warp_mma_k) {
        warp_tile_iterator_A.load(warp_tile_frag_A);
        warp_tile_iterator_B.load(warp_tile_frag_B);

        warp_mma(accum, warp_tile_frag_A, warp_tile_frag_B, accum);

        ++warp_tile_iterator_A;
        ++warp_tile_iterator_B;
      }

      warp_tile_iterator_A.reset();
      warp_tile_iterator_B.reset();

      iterator_A.load(tb_frag_A);
      iterator_B.load(tb_frag_B);

      ++iterator_A;
      ++iterator_B;

      // Avoid reading out of bounds
      iterator_A.clear_mask(gemm_k_iterations <= 2);
      iterator_B.clear_mask(gemm_k_iterations <= 2);
    }
  }

};

}
}
}