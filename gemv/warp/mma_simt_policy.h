#pragma once


#include "cutlass/cutlass.h"
#include "cutlass/gemm_coord.h"

namespace cutlass {
namespace gemm {
namespace warp {

template<
 /// Concept: WarpThreadArrangement
 typename WarpShape_,
 typename MmaShape_ = cutlass::gemm::GemmShape<1, 1, 1>
>
class MmaSimtGemvPolicy {
public:
  using WarpShape = WarpShape_;
  using MmaShape = MmaShape_;
};

/// Policy object describing MmaTensorOp
template <
    /// Warp-level GEMM operator (concept: gemm::warp::Mma)
    typename Operator_,
    int WarpPartitionsK = 1
>
struct MmaGemvPolicy {
  /// Warp-level GEMM operator (concept: gemm::warp::MmaTensorOp or gemm::warp::MmaSimt)
  using Operator = Operator_;
  static const int kWarpPartitionsK = WarpPartitionsK;
};

} /// end of namespace warp
} /// end of namespace gemm
} /// end of namespace cutlass