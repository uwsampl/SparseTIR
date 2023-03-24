/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \brief tvm/tir/sparse.h
 * \brief sparse axes and buffers.
 */
#ifndef TVM_TIR_SPARSE_H_
#define TVM_TIR_SPARSE_H_

#include <tvm/ir/expr.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/string.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/var.h>

namespace tvm {
namespace tir {

enum class AxisKind : int {
  kDenseFixed = 0,
  kDenseVariable = 1,
  kSparseFixed = 2,
  kSparseVariable = 3
};

/*!
 * \brief Base type for axis in sparse formats.
 */
class AxisNode : public Object {
 public:
  /* name of current axis. */
  String name;
  /* The parent axis. */
  Optional<ObjectRef> parent;
  /* length of current axis. For sparse axis, length refers to the upperbound of
   * the current axis. */
  PrimExpr length;
  /* The accumulated number of nonzero elements from root axis to current axis. */
  PrimExpr nnz;
  /* The number of nonzero columns in current row, only valid for fixed axis. */
  Optional<PrimExpr> nnz_cols;
  /* The indptr buffer var. */
  Optional<Var> indptr;
  /* The indices buffer var. */
  Optional<Var> indices;
  /* The index data type. */
  DataType idtype;
  /* Whether the indices are sorted or not. */
  bool sorted;

  /* Whether current axis is a variable axis. */
  bool IsVariable() const { return indptr.defined(); }

  /* Whether current axis is a sparse axis. */
  bool IsSparse() const { return indices.defined(); }

  AxisKind kind() const {
    return AxisKind(static_cast<int>(IsVariable()) + 2 * static_cast<int>(IsSparse()));
  }

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("length", &length);
    v->Visit("nnz", &nnz);
    v->Visit("nnz_cols", &nnz_cols);
    v->Visit("parent", &parent);
    v->Visit("indptr", &indptr);
    v->Visit("indices", &indices);
    v->Visit("idtype", &idtype);
    v->Visit("sorted", &sorted);
  }

  bool SEqualReduce(const AxisNode* other, SEqualReducer equal) const {
    return equal(name, other->name) && equal(length, other->length) && equal(nnz, other->nnz) &&
           equal(nnz_cols, other->nnz_cols) && equal(parent, other->parent) &&
           equal(indptr, other->indptr) && equal(indices, other->indices) &&
           equal(idtype, other->idtype) && equal(sorted, other->sorted);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(name);
    hash_reduce(length);
    hash_reduce(nnz);
    hash_reduce(nnz_cols);
    hash_reduce(parent);
    hash_reduce(indptr);
    hash_reduce(indices);
    hash_reduce(idtype);
    hash_reduce(sorted);
  }

  static constexpr const char* _type_key = "tir.sparse.Axis";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_BASE_OBJECT_INFO(AxisNode, Object);
};

/*!
 * \brief Managed reference to AxisNode.
 * \sa AxisNode
 */
class Axis : public ObjectRef {
 public:
  TVM_DLL explicit Axis(String name, Optional<ObjectRef> parent, PrimExpr length, PrimExpr nnz,
                        Optional<PrimExpr> nnz_cols, Optional<Var> indptr, Optional<Var> indices,
                        DataType idtype, bool sorted = true);

  TVM_DEFINE_OBJECT_REF_METHODS(Axis, ObjectRef, AxisNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(AxisNode);
};

Axis GetParentAxis(const Axis& axis);

Axis GetRootAxis(const Axis& axis);

/*! \brief Derivation axis, constructed by T.fuse(axis1, axis2, ...) */
class FusedAxisNode : public AxisNode {
 public:
  /* The group of axes to be fused. */
  Array<Axis> group;
  /* The index of current FusedAxis in the group. */
  int index;

  void VisitAttrs(AttrVisitor* v) {
    AxisNode::VisitAttrs(v);
    v->Visit("group", &group);
    v->Visit("index", &index);
  }

  bool IsLastAxis() const { return index + 1 == static_cast<int>(group.size()); }

  bool SEqualReduce(const FusedAxisNode* other, SEqualReducer equal) const {
    return AxisNode::SEqualReduce(other, equal) && equal(group, other->group) &&
           equal(index, other->index);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    AxisNode::SHashReduce(hash_reduce);
    hash_reduce(group);
    hash_reduce(index);
  }

  static constexpr const char* _type_key = "tir.sparse.FusedAxis";
  TVM_DECLARE_FINAL_OBJECT_INFO(FusedAxisNode, AxisNode);
};

/*!
 * \brief Managed reference to FusedAxisNode.
 * \sa FusedAxisNode
 */
class FusedAxis : public Axis {
 public:
  /* Fused axis could be constructed by specifying a group of based axes and an index */
  TVM_DLL explicit FusedAxis(Array<Axis> group, int index);

  TVM_DEFINE_OBJECT_REF_METHODS(FusedAxis, Axis, FusedAxisNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(FusedAxisNode);
};

/*!
 * \brief Flatten several axes for compact storage.
 */
class FlattenedAxisNode : public AxisNode {
 public:
  /* The axes to flatten. */
  Array<Axis> axes;
  /* The number of nonzero elements in flattened axis. */
  PrimExpr flattened_nnz;
  /* The offset array */
  Buffer offset;

  void VisitAttrs(AttrVisitor* v) {
    AxisNode::VisitAttrs(v);
    v->Visit("axes", &axes);
  }

  bool SEqualReduce(const FlattenedAxisNode* other, SEqualReducer equal) const {
    return AxisNode::SEqualReduce(other, equal) && equal(axes, other->axes);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    AxisNode::SHashReduce(hash_reduce);
    hash_reduce(axes);
  }

  static constexpr const char* _type_key = "tir.sparse.FlattenedAxis";
  TVM_DECLARE_FINAL_OBJECT_INFO(FlattenedAxisNode, AxisNode);
};

/*!
 * \brief Managed reference to FlattenedAxisNode.
 * \sa FlattenedAxisNode
 */
class FlattenedAxis : public Axis {
 public:
  TVM_DLL explicit FlattenedAxis(String name, Array<Axis> axes, PrimExpr flattened_nnz,
                                 Buffer offset);
  TVM_DEFINE_OBJECT_REF_METHODS(FlattenedAxis, Axis, FlattenedAxisNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(FlattenedAxisNode);
};

/* Return a new dense axis that inherits the dependency of base axis.*/
Axis ToDenseAxis(Axis base);

/*!
 * \brief Attach an axis to a new parent, constructed by T.attach.
 */
class AttachedAxisNode : public AxisNode {
 public:
  /*! \brief The based sparse axis. */
  Axis base;

  void VisitAttrs(AttrVisitor* v) {
    AxisNode::VisitAttrs(v);
    v->Visit("base", &base);
  }

  bool SEqualReduce(const AttachedAxisNode* other, SEqualReducer equal) const {
    return AxisNode::SEqualReduce(other, equal) && equal(base, other->base);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    AxisNode::SHashReduce(hash_reduce);
    hash_reduce(base);
  }

  static constexpr const char* _type_key = "tir.sparse.AttachedAxis";
  TVM_DECLARE_FINAL_OBJECT_INFO(AttachedAxisNode, AxisNode);
};

/*!
 * \brief Managed reference to AttachedAxis.
 * \sa AttachedAxisNode
 */
class AttachedAxis : public Axis {
 public:
  TVM_DLL explicit AttachedAxis(Axis base, Axis new_parent);

  TVM_DEFINE_OBJECT_REF_METHODS(AttachedAxis, Axis, AttachedAxisNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(AttachedAxisNode);
};

/*! \brief Node to represent a sparse buffer */
class SparseBufferNode : public BufferNode {
 public:
  // Data fields.
  /*!
   * \brief The axes used in the sparse buffer.
   */
  Array<Axis> axes;

  Optional<PrimExpr> extra_storage;

  PrimExpr GetNNZ() const;

  Buffer flattened;
  /*!
   * \brief The default value in the sparse buffer.
   */
  Optional<PrimExpr> default_value;
  void VisitAttrs(AttrVisitor* v) {
    BufferNode::VisitAttrs(v);
    v->Visit("axes", &axes);
    v->Visit("extra_storage", &extra_storage);
    v->Visit("flattened", &flattened);
    v->Visit("default_value", &default_value);
  }

  bool SEqualReduce(const SparseBufferNode* other, SEqualReducer equal) const {
    return BufferNode::SEqualReduce(other, equal) && equal(axes, other->axes) &&
           equal(extra_storage, other->extra_storage) && equal(flattened, other->flattened) &&
           equal(default_value, other->default_value);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    BufferNode::SHashReduce(hash_reduce);
    hash_reduce(axes);
    hash_reduce(extra_storage);
    hash_reduce(flattened);
    hash_reduce(default_value);
  }

  static constexpr const char* _type_key = "tir.sparse.SparseBuffer";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(SparseBufferNode, BufferNode);
};

/*!
 * \brief Managed reference to SparseBufferNode.
 * \sa SparseBufferNode
 */
class SparseBuffer : public Buffer {
 public:
  TVM_DLL explicit SparseBuffer(Var data, Array<Axis> axes, DataType dtype, String name,
                                Optional<PrimExpr> extra_storage,
                                Optional<PrimExpr> default_value = NullOpt, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(SparseBuffer, Buffer, SparseBufferNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(SparseBufferNode);
};

// overload printing of for type.
TVM_DLL std::ostream& operator<<(std::ostream& os, AxisKind kind);

/*!
 * \brief Iterator variables in SparseTIR
 */
class SpIterVarNode : public Object {
 public:
  Axis axis;
  Var var;
  bool is_reduction;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("axis", &axis);
    v->Visit("is_reduction", &is_reduction);
    v->Visit("var", &var);
  }

  bool SEqualReduce(const SpIterVarNode* other, SEqualReducer equal) const {
    return equal(axis, other->axis) && equal.DefEqual(var, other->var) &&
           equal(is_reduction, other->is_reduction);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(axis);
    hash_reduce(var);
    hash_reduce(is_reduction);
  }

  IterVar as_iter_var() const {
    tvm::Range dom;
    if (axis->IsVariable()) {
      // if variable axis, set dom to maximum length
      dom = Range::FromMinExtent(Integer(0), axis->length);
    } else {
      // if not variable axis, set dom to the number of non zero columns.
      dom = Range::FromMinExtent(Integer(0), axis->nnz_cols.value());
    }

    return IterVar(dom, Var("v" + var->name_hint, var->dtype),
                   is_reduction ? kCommReduce : kDataPar, "");
  }

  static constexpr const char* _type_key = "tir.sparse.SpIterVar";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(SpIterVarNode, IterVarNode);
};

class SpIterVar : public ObjectRef {
 public:
  TVM_DLL explicit SpIterVar(Var var, bool is_reduction, Axis axis);

  TVM_DEFINE_OBJECT_REF_METHODS(SpIterVar, ObjectRef, SpIterVarNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(SpIterVarNode);
};

// inline implementations
inline const char* SpIterKind2String(AxisKind t) {
  switch (t) {
    case AxisKind::kDenseFixed:
      return "dense_fixed";
    case AxisKind::kDenseVariable:
      return "dense_variable";
    case AxisKind::kSparseFixed:
      return "sparse_fixed";
    case AxisKind::kSparseVariable:
      return "sparse_variable";
    default:
      throw;
  }
}

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SPARSE_H_
