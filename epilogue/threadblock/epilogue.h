#pragma once


#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/layout/vector.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/aligned_buffer.h"
#include "cutlass/functional.h"


namespace cutlass {
namespace epilogue {
namespace threadblock {

template<
 typename Shape_,
 typename ElementAccumulator_,
 typename OutputTileIterator_,
 typename OutputOp_
>
class EpilogueGemv {
public:
 using Shape = Shape_;
 using ElementAccumulator = ElementAccumulator_;
 using OutputTileIterator = OutputTileIterator_;
 using OutputOp = OutputOp_;
 using Layout = layout::RowMajor;

 static const int kElementsPerAccess = OutputTileIterator::AccessType::kElements;

 using AccumulatorFragment = cutlass::Array<ElementAccumulator, OutputTileIterator::Fragment::kElements>;
 using AccumulatorAccessType = cutlass::Array<ElementAccumulator, kElementsPerAccess>;
 using OutputAccessType = Array<typename OutputTileIterator::Element, kElementsPerAccess>;

 static const int kOutputOpIterations = OutputTileIterator::Fragment::kElements / kElementsPerAccess;

 struct SourceNeeded {
  OutputTileIterator source_iterator;
  typename OutputTileIterator::Fragment source_fragment;

 public:
  CUTLASS_DEVICE
  SourceNeeded(OutputTileIterator source_iterator_) : source_iterator(source_iterator_) {
    source_fragment.clear();
  };

  CUTLASS_DEVICE
  static void run(
      typename OutputTileIterator::Fragment& output_fragment,
      const OutputOp& output_op,
      const AccumulatorFragment &accum_fragment,
      const typename OutputTileIterator::Fragment& source_fragment
  ) {
    const AccumulatorAccessType *acc_frag_ptr = reinterpret_cast<const AccumulatorAccessType*>(&accum_fragment);
    OutputAccessType *output_frag_ptr = reinterpret_cast<OutputAccessType *>(&output_fragment);

    const OutputAccessType *source_frag_ptr = reinterpret_cast<const OutputAccessType*>(&source_fragment);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kOutputOpIterations; ++i) {
      output_frag_ptr[i] = output_op(acc_frag_ptr[i], source_frag_ptr[i]);
    }
  }

  CUTLASS_DEVICE
  void apply_output_operator(typename OutputTileIterator::Fragment &output_fragment,
                             const OutputOp &output_op,
                             const AccumulatorFragment &accum_fragment) {
    source_iterator.load(source_fragment);
    ++source_iterator;
    run(output_fragment, output_op, accum_fragment, source_fragment);
  }
 };

 struct SourceNotNeeded {
 public:
  CUTLASS_DEVICE
  SourceNotNeeded() {}

  CUTLASS_DEVICE
  void apply_output_operator(typename OutputTileIterator::Fragment &output_fragment,
                             const OutputOp &output_op,
                             const AccumulatorFragment &accum_fragment) {
    const AccumulatorAccessType *acc_frag_ptr =
        reinterpret_cast<const AccumulatorAccessType *>(&accum_fragment);
    OutputAccessType *output_frag_ptr = reinterpret_cast<OutputAccessType *>(&output_fragment);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kOutputOpIterations; ++i) {
      output_frag_ptr[i] = output_op(acc_frag_ptr[i]);
    }
  }
 };

 CUTLASS_HOST_DEVICE
 EpilogueGemv() {}

public:
  template <typename Source>
  CUTLASS_DEVICE
  void run(
    const OutputOp& output_op,
    OutputTileIterator destination_iterator,
    AccumulatorFragment& accumulators,
    Source source
  ) {

    typename OutputTileIterator::Fragment output_fragment;
    output_fragment.clear();

    source.apply_output_operator(output_fragment, output_op, accumulators);

    // int warp_idx = threadIdx.x / 32;
    // int lane_idx = threadIdx.x % 32;

    // if (blockIdx.x == 0 && warp_idx == 0 && lane_idx == 4) {
    //   for (int i = 0; i < output_fragment.size(); i++) {
    //     printf("block %d, warp %d, lane %d, output_fragment[%d]=%f\n",
    //            blockIdx.x,
    //            warp_idx,
    //            lane_idx,
    //            i,
    //            output_fragment[i]);
    //   }
    // }

    destination_iterator.store(output_fragment);
  }


  CUTLASS_DEVICE
  void operator()(const OutputOp &output_op,
                  OutputTileIterator destination_iterator,
                  AccumulatorFragment &accumulators,
                  OutputTileIterator source_iterator) {
    if (output_op.is_source_needed()) {
      run(output_op, destination_iterator, accumulators, SourceNeeded(source_iterator));
    } else {
      run(output_op, destination_iterator, accumulators, SourceNotNeeded());
    }
  }
};

} /// end of namespace threadblock
} /// end of namespace epilogue
} /// end of namespace cutlass