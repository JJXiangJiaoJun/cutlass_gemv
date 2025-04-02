#pragma once


#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/layout/matrix.h"

namespace cutlass {
namespace gemm {
namespace warp {

template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  /// Operand identity
  // Operand Operand,
  typename RegisteTileShape_,
  /// Data type of A elements
  typename Element_,
  // /// Layout of operand
  // typename Layout_,
  /// Shape of the warp in units of thread (concept: MmaSimtPolicy)
  typename Policy_,

  ////
  typename Layout_ = cutlass::layout::RowMajor
>
class MmaSimtRegisterTileIterator {
public:
  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  using RegisteTileShape = RegisteTileShape_;

  using Policy = Policy_;

  /// Element type
  using Element = Element_;

  using ThreadShape = cutlass::MatrixShape<Shape::kRow / Policy::WarpShape::kRow, Shape::kColumn>;

  using FragmentRegisterTile = cutlass::Array<Element, RegisteTileShape::kCount>;

  static_assert(RegisteTileShape::kRow == ThreadShape::kRow, "");

  using Fragment = cutlass::Array<Element, ThreadShape::kCount>;

  using AccessType = cutlass::Array<Element, ThreadShape::kColumn>;

  using TensorCoord = cutlass::MatrixCoord;

private:
  AccessType * pointer_;
  int row_    = 0;
  int column_ = 0;

public:
  CUTLASS_HOST_DEVICE
  MmaSimtRegisterTileIterator() {}

  CUTLASS_DEVICE
  MmaSimtRegisterTileIterator(FragmentRegisterTile &frag_register) {
    pointer_ = reinterpret_cast<AccessType *>(&frag_register);
    row_     = 0;
    column_  = 0;
  }

  CUTLASS_DEVICE
  void set_register_tile(FragmentRegisterTile &frag_register) {
    pointer_ = reinterpret_cast<AccessType *>(&frag_register);
  }

  CUTLASS_DEVICE
  void reset() {
    row_     = 0;
    column_  = 0;
  }

  CUTLASS_DEVICE
  void add_tile_offset(const TensorCoord& tile_offset) {
    row_ += tile_offset.row();
    column_ += tile_offset.column();
  }

  CUTLASS_DEVICE
  MmaSimtRegisterTileIterator& operator++() {
    add_tile_offset({0, 1});
    return *this;
  }

  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) {
    AccessType *ptr_frag = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < ThreadShape::kRow; r++) {
      ptr_frag[r] = pointer_[r * RegisteTileShape::kColumn + column_];
    }

  }

};

} /// end of namespace warp
} /// end of namespace gemm
} /// end of namespace cutlass