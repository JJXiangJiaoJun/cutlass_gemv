#pragma once


#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/gemm_enumerated_types.h"

namespace cutlass {
namespace gemm {
namespace warp {

template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  /// Operand identity
  typename RegisterTileShape_,
  /// Operand identity
  cutlass::gemm::Operand OperandType,
  /// Data type of A elements
  typename Element_,
  // /// Layout of operand
  // typename Layout_,
  /// Shape of the warp in units of thread (concept: MmaSimtPolicy)
  typename Policy,
  //// Accumulator register layout
  typename Layout
>
class MmaSimtRegisterTileIterator;

template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  // Operand Operand,
  typename RegisterTileShape_,
  /// Data type of A elements
  typename Element_,
  // /// Layout of operand
  // typename Layout_,
  /// Shape of the warp in units of thread (concept: MmaSimtPolicy)
  typename Policy_
>
class MmaSimtRegisterTileIterator<
  Shape_,
  RegisterTileShape_,
  Operand::kA,
  Element_,
  Policy_,
  cutlass::layout::RowMajor
> {
public:
  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  using RegisterTileShape = RegisterTileShape_;

  using Policy = Policy_;

  /// Element type
  using Element = Element_;

  using ThreadShape = cutlass::MatrixShape<Shape::kRow / Policy::WarpShape::kRow, Shape::kColumn>;

  using FragmentRegisterTile = cutlass::Array<Element, RegisterTileShape::kCount>;

  static_assert(RegisterTileShape::kRow == ThreadShape::kRow, "");

  using Fragment = cutlass::Array<Element, ThreadShape::kCount>;

  using AccessType = cutlass::Array<Element, ThreadShape::kColumn>;

  using TensorCoord = cutlass::MatrixCoord;

  static const cutlass::gemm::Operand kOperand = Operand::kA;

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
    column_ += tile_offset.column() * ThreadShape::kColumn;
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
    for (int r = 0; r < ThreadShape::kRow; ++r) {
      ptr_frag[r] = pointer_[r * RegisterTileShape::kColumn / ThreadShape::kColumn  + column_];
    }

  }

};

template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  // Operand Operand,
  typename RegisterTileShape_,
  /// Data type of A elements
  typename Element_,
  // /// Layout of operand
  // typename Layout_,
  /// Shape of the warp in units of thread (concept: MmaSimtPolicy)
  typename Policy_
>
class MmaSimtRegisterTileIterator<
  Shape_,
  RegisterTileShape_,
  Operand::kB,
  Element_,
  Policy_,
  cutlass::layout::ColumnMajor
> {
public:
  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  using RegisterTileShape = RegisterTileShape_;

  using Policy = Policy_;

  /// Element type
  using Element = Element_;

  using ThreadShape = cutlass::MatrixShape<Shape::kRow / Policy::WarpShape::kRow, Shape::kColumn>;

  using FragmentRegisterTile = cutlass::Array<Element, RegisterTileShape::kCount>;

  static_assert(RegisterTileShape::kRow == ThreadShape::kRow, "");

  using Fragment = cutlass::Array<Element, ThreadShape::kCount>;

  using AccessType = cutlass::Array<Element, ThreadShape::kColumn>;

  using TensorCoord = cutlass::MatrixCoord;

  static const cutlass::gemm::Operand kOperand = Operand::kB;

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
    column_ += tile_offset.column() * ThreadShape::kColumn;
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
    for (int r = 0; r < ThreadShape::kRow; ++r) {
      ptr_frag[r] = pointer_[r * RegisterTileShape::kColumn / ThreadShape::kColumn + column_];
    }

  }

};

template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  // Accumulator register tile shape (concept: MatrixShape)
  typename RegisterTileShape_,
  /// Data type of A elements
  typename Element_,
  // /// Layout of operand
  // typename Layout_,
  /// Shape of the warp in units of thread (concept: MmaSimtPolicy)
  typename Policy_
>
class MmaSimtRegisterTileIterator<
  Shape_,
  RegisterTileShape_,
  Operand::kA,
  Element_,
  Policy_,
  cutlass::layout::ColumnMajor
> {
public:
  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  using RegisterTileShape = RegisterTileShape_;

  using Policy = Policy_;

  /// Element type
  using Element = Element_;

  using ThreadShape = cutlass::MatrixShape<Shape::kRow / Policy::WarpShape::kRow, Shape::kColumn>;

  using FragmentRegisterTile = cutlass::Array<Element, RegisterTileShape::kCount>;

  static_assert(RegisterTileShape::kRow == ThreadShape::kRow, "");

  using Fragment = cutlass::Array<Element, ThreadShape::kCount>;

  using AccessType = cutlass::Array<Element, ThreadShape::kRow>;

  using TensorCoord = cutlass::MatrixCoord;

  static const cutlass::gemm::Operand kOperand = Operand::kA;

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
    column_ += tile_offset.column() * ThreadShape::kColumn;
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
    for (int c = 0; c < ThreadShape::kColumn; ++c) {
      ptr_frag[c] = pointer_[(column_ + c) * (RegisterTileShape::kRow / ThreadShape::kRow)];
    }

  }

};


template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  // Accumulator register tile shape (concept: MatrixShape)
  typename RegisterTileShape_,
  /// Data type of A elements
  typename Element_,
  // /// Layout of operand
  // typename Layout_,
  /// Shape of the warp in units of thread (concept: MmaSimtPolicy)
  typename Policy_
>
class MmaSimtRegisterTileIterator<
  Shape_,
  RegisterTileShape_,
  Operand::kB,
  Element_,
  Policy_,
  cutlass::layout::RowMajor
> {
public:
  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  using RegisterTileShape = RegisterTileShape_;

  using Policy = Policy_;

  /// Element type
  using Element = Element_;

  using ThreadShape = cutlass::MatrixShape<Shape::kRow / Policy::WarpShape::kRow, Shape::kColumn>;

  using FragmentRegisterTile = cutlass::Array<Element, RegisterTileShape::kCount>;

  static_assert(RegisterTileShape::kRow == ThreadShape::kRow, "");

  using Fragment = cutlass::Array<Element, ThreadShape::kCount>;

  using AccessType = cutlass::Array<Element, ThreadShape::kRow>;

  using TensorCoord = cutlass::MatrixCoord;

  static const cutlass::gemm::Operand kOperand = Operand::kA;

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
    column_ += tile_offset.column() * ThreadShape::kColumn;
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
    for (int c = 0; c < ThreadShape::kColumn; ++c) {
      ptr_frag[c] = pointer_[(column_ + c) * (RegisterTileShape::kRow / ThreadShape::kRow)];
    }

  }

};

} /// end of namespace warp
} /// end of namespace gemm
} /// end of namespace cutlass