#pragma once


#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/arch/mma.h"

namespace cutlass {
namespace gemm {
namespace warp {

template<
  typename Element_,
  int Count,
  int ThreadsPerGroup
>
class Reduce {
public:

  using Element = Element_;
  using Fragment = cutlass::Array<Element, Count>;
  static const int kCount = Count;
  static const int kThreadsPerGroup = ThreadsPerGroup;

  CUTLASS_HOST_DEVICE
  Reduce() {}

  CUTLASS_DEVICE
  void operator()(Fragment &frag) {

    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < kCount; k++) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = kThreadsPerGroup / 2; i >= 1; i >>= 1) {
        *(frag.data() + k) += __shfl_down_sync(0xffffffff, *(frag.data() + k), i, kThreadsPerGroup);
      }
    }
  }

};

} /// end of namespace warp
} /// end of namespace gemm
} /// end of namespace cutlass