#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/tensor_ref.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/device_kernel.h"

#include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "gemv/kernel/default_gemv.h"

namespace cutlass {
namespace gemm {
namespace device {

template <
  /// Element type for A matrix operand
  typename ElementA_,
  /// Layout type for A matrix operand
  typename LayoutA_,
  /// Element type for B matrix operand
  typename ElementB_,
  /// Layout type for B matrix operand
  typename LayoutB_,
  /// Element type for C and D matrix operands
  typename ElementC_,
  /// Layout type for C and D matrix operands
  typename LayoutC_,
  /// Element type for internal accumulation
  typename ElementAccumulator_,
  /// Operator class tag
  typename OperatorClass_,
  /// Tag indicating architecture to tune for
  typename ArchTag_,
  /// Threadblock-level tile size (concept: GemmShape)
  typename ThreadBlockShape_,
  /// Warp-level tile size (concept: GemmShape)
  typename WarpShape_,
  /// Instruction-level tile size (concept: GemmShape)
  typename InstructionShape_,
  /// Warp thread layout (concept: MatrixShape)
  typename WarpThreadArrangement_,
  /// Number of stages used in the pipelined mainloop
  typename EpilogueOutputOp_,
  int Stages,
  int AlignmentA,
  int AlignmentB
>
class GemvAdaptor {
public:
  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using TensorRefA = TensorRef<ElementA const, LayoutA>;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using TensorRefB = TensorRef<ElementB const, LayoutB>;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using TensorRefC = TensorRef<ElementC const, LayoutC>;
  using TensorRefD = TensorRef<ElementC, LayoutC>;
  using ElementAccumulator = ElementAccumulator_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadBlockShape = ThreadBlockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using WarpThreadArrangement = WarpThreadArrangement_;
  using EpilogueOutputOp = EpilogueOutputOp_;

  static int const kStages = Stages;
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;

  using GemvKernel = typename cutlass::gemm::kernel::DefaultGemv<ElementA,
                                                                 LayoutA,
                                                                 kAlignmentA,
                                                                 ElementB,
                                                                 LayoutB,
                                                                 kAlignmentB,
                                                                 ElementC,
                                                                 LayoutC,
                                                                 ElementAccumulator,
                                                                 OperatorClass,
                                                                 ArchTag,
                                                                 ThreadBlockShape,
                                                                 WarpShape,
                                                                 InstructionShape,
                                                                 WarpThreadArrangement,
                                                                 EpilogueOutputOp>::Kernel;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    GemmCoord problem_size;
    TensorRef<ElementA const, LayoutA> ref_A;
    TensorRef<ElementB const, LayoutB> ref_B;
    TensorRef<ElementC const, LayoutC> ref_C;
    TensorRef<ElementC, LayoutC> ref_D;
    typename EpilogueOutputOp::Params epilogue;

    //
    // Methods
    //

    /// Constructs an Arguments structure
    CUTLASS_HOST_DEVICE
    Arguments(
      GemmCoord problem_size_,
      TensorRef<ElementA const, LayoutA> ref_A_,
      TensorRef<ElementB const, LayoutB> ref_B_,
      TensorRef<ElementC const, LayoutC> ref_C_,
      TensorRef<ElementC, LayoutC> ref_D_,
      typename EpilogueOutputOp::Params epilogue_ =
        typename EpilogueOutputOp::Params()
      // int split_k_slices = 1,
      // int const *gather_A_indices_ = nullptr,
      // int const *gather_B_indices_ = nullptr,
      // int const *scatter_D_indices_ = nullptr
    ):
      problem_size(problem_size_),
      ref_A(ref_A_),
      ref_B(ref_B_),
      ref_C(ref_C_),
      ref_D(ref_D_),
      epilogue(epilogue_) {

    }
  };

private:

  /// Kernel parameters object
  typename GemvKernel::Params params_;

public:

  CUTLASS_HOST_DEVICE
  GemvAdaptor() {}

  static dim3 get_grid_size(typename GemvKernel::Params &params) {
    return dim3((params.problem_size.m() + GemvKernel::Mma::Shape::kM - 1) / GemvKernel::Mma::Shape::kM, 1, 1);
  }

  static dim3 get_block_size(typename GemvKernel::Params &params) {
    return dim3(GemvKernel::kThreads, 1, 1);
  }

  void initialize(Arguments const &args, cudaStream_t stream = nullptr) {
    // Initialize the Params structure
    params_ = typename GemvKernel::Params{
      args.problem_size,
      args.ref_A.non_const_ref(),
      args.ref_B.non_const_ref(),
      args.ref_C.non_const_ref(),
      args.ref_D,
      args.epilogue
    };
  }

  /// Runs the kernel using initialized state.
  void run(cudaStream_t stream = nullptr) {

    dim3 grid = get_grid_size(params_);
    dim3 block = get_block_size(params_);

    cudaError_t result;

    int smem_size = int(sizeof(typename GemvKernel::SharedStorage));

    if (smem_size >= (48 << 10)) {
      result = cudaFuncSetAttribute(Kernel<GemvKernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (result != cudaSuccess) {
        std::cout << "failed"  << std::endl;
        return;
      }
    }

    cutlass::Kernel<GemvKernel><<<grid, block, smem_size, stream>>>(params_);

    result = cudaGetLastError();

    if (result != cudaSuccess) {
      std::cout << "launch failed" << std::endl;
      return;
    }
  }

  /// Runs the kernel using initialized state.
  void operator()(Arguments const &args, cudaStream_t stream = nullptr) {
    initialize(args);
    run(stream);
  }
};

}
}
}