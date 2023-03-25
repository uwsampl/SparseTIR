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
 * \file sparse.cc
 * \brief buffers and formats in sparse tir.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/format_rewrite.h>
#include <tvm/tir/op.h>
#include <tvm/tir/sparse.h>

namespace tvm {
namespace tir {

/******** Axis utility functions ********/

Axis GetParentAxis(const Axis& axis) {
  Optional<ObjectRef> parent_obj = axis->parent;
  CHECK(parent_obj.defined()) << "Parent axis is not defined.";
  return Downcast<Axis>(parent_obj.value());
}

Axis GetRootAxis(const Axis& axis) {
  // NOTE(zihao): for fixed axis, return itself whether we have parent axis.
  if (!axis->IsVariable()) {
    return axis;
  }
  Optional<ObjectRef> parent_obj = axis->parent;
  if (parent_obj.defined()) {
    Axis parent = Downcast<Axis>(parent_obj.value());
    return GetRootAxis(parent);
  } else {
    return axis;
  }
}

/******** Axis ********/

/*! \brief Default constructor of Axis */
Axis::Axis(String name, Optional<ObjectRef> parent, PrimExpr length, PrimExpr nnz,
           Optional<PrimExpr> nnz_cols, Optional<Var> indptr, Optional<Var> indices,
           DataType idtype, bool sorted) {
  ObjectPtr<AxisNode> node = make_object<AxisNode>();
  node->name = std::move(name);
  node->length = std::move(length);
  node->parent = std::move(parent);
  node->nnz = std::move(nnz);
  node->nnz_cols = std::move(nnz_cols);
  node->indptr = std::move(indptr);
  node->indices = std::move(indices);
  node->idtype = std::move(idtype);
  node->sorted = sorted;
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(AxisNode);

TVM_REGISTER_GLOBAL("tir.sparse.Axis")
    .set_body_typed([](String name, Optional<Axis> parent, PrimExpr length, PrimExpr nnz,
                       Optional<PrimExpr> nnz_cols, Optional<Var> indptr, Optional<Var> indices,
                       DataType idtype, Bool sorted) {
      return Axis(std::move(name), std::move(parent), std::move(length), std::move(nnz),
                  std::move(nnz_cols), std::move(indptr), std::move(indices), std::move(idtype),
                  sorted->value);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<AxisNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const AxisNode*>(node.get());
      switch (op->kind()) {
        case AxisKind::kDenseFixed:
          p->stream << "dense_fixed(" << op->name << ", " << op->length << ")";
          break;
        case AxisKind::kDenseVariable:
          p->stream << "dense_variable(" << op->name << ", " << op->length << ", "
                    << op->indptr.value() << ")";
          break;
        case AxisKind::kSparseFixed:
          p->stream << "sparse_fixed(" << op->name << ", "
                    << Downcast<Axis>(op->parent.value())->name << ", " << op->length << ", "
                    << op->nnz_cols << ", " << op->indices.value() << ")";
          break;
        case AxisKind::kSparseVariable:
          p->stream << "sparse_variable(" << op->name << ", " << op->length << ", "
                    << op->indptr.value() << ", " << op->indices.value() << ")";
          break;
        default:
          throw;
      }
    });

/******** FusedAxis ********/

/*! \brief Default constructor of FusedAxis */
FusedAxis::FusedAxis(Array<Axis> group, int index) {
  CHECK(index < static_cast<int>(group.size()))
      << "Index " << index << "exceeds the size of fused axes group.";

  // TODO(zihao): check whether it valid to fuse axes in the group.
  ObjectPtr<FusedAxisNode> node = make_object<FusedAxisNode>();
  node->group = std::move(group);
  node->index = index;
  std::string fused_name = node->group[0]->name;
  for (size_t i = 1; i < node->group.size(); ++i) {
    fused_name += node->group[i]->name;
  }
  node->name = "fused_" + fused_name + "_" + node->group[index]->name;
  if (index == static_cast<int>(node->group.size()) - 1) {
    // is last fused axis.
    node->length = node->group[index]->nnz;
  } else {
    node->length = Integer(1);
  }
  node->nnz = node->length;
  node->nnz_cols = node->length;
  node->indptr = NullOpt;
  node->indices = NullOpt;
  node->idtype = node->group[0]->idtype;
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(FusedAxisNode);

TVM_REGISTER_GLOBAL("tir.sparse.FusedAxis").set_body_typed([](Array<Axis> group, int index) {
  return FusedAxis(std::move(group), index);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<FusedAxisNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const FusedAxisNode*>(node.get());
      p->stream << "fused(";
      bool first = true;
      for (auto&& orig_axis : op->group) {
        if (first) {
          first = false;
        } else {
          p->stream << ", ";
        }
        p->stream << orig_axis->name;
      }
      p->stream << ")";
    });

TVM_REGISTER_GLOBAL("tir.sparse.GetFusedAxisGroup").set_body_typed([](FusedAxis axis) {
  return axis->group;
});

TVM_REGISTER_GLOBAL("tir.sparse.GetFusedAxisIndex").set_body_typed([](FusedAxis axis) {
  return axis->index;
});

/******** ToDenseAxis ********/

Axis ToDenseAxis(Axis base) {
  return Axis(base->name + "_dense", base->IsVariable() ? base->parent : NullOpt,
              base->IsVariable() ? base->length : base->nnz_cols.value(),
              base->IsVariable() ? base->nnz : base->nnz_cols.value(), base->nnz_cols, base->indptr,
              NullOpt, base->idtype);
}

/******** FlattenedAxis ********/
FlattenedAxis::FlattenedAxis(String name, Array<Axis> axes, PrimExpr flattened_nnz, Buffer offset) {
  ObjectPtr<FlattenedAxisNode> node = make_object<FlattenedAxisNode>();
  node->axes = std::move(axes);
  node->flattened_nnz = std::move(flattened_nnz);
  node->offset = std::move(offset);
  node->name = std::move(name);
  node->length = node->axes[0]->length;
  node->nnz = node->axes[0]->nnz;
  node->nnz_cols = node->axes[0]->nnz_cols;
  node->indptr = node->axes[0]->indptr;
  node->indices = node->axes[0]->indices;
  node->idtype = node->axes[0]->idtype;
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(FlattenedAxisNode);

TVM_REGISTER_GLOBAL("tir.sparse.FlattenedAxis")
    .set_body_typed([](String name, Array<Axis> axes, PrimExpr flattened_nnz, Buffer offset) {
      return FlattenedAxis(std::move(name), std::move(axes), std::move(flattened_nnz),
                           std::move(offset));
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<FlattenedAxisNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const FlattenedAxisNode*>(node.get());
      p->stream << "flatten(" << op->name << ", " << op->axes << ")";
    });

/******** AttachedAxis ********/
AttachedAxis::AttachedAxis(Axis base, Axis new_parent) {
  ObjectPtr<AttachedAxisNode> node = make_object<AttachedAxisNode>();
  node->base = std::move(base);
  node->name = node->base->name + "_attach_" + new_parent->name;
  node->parent = std::move(new_parent);
  node->length = node->base->length;
  node->nnz = node->base->nnz;
  node->nnz_cols = node->base->nnz_cols;
  node->indptr = node->base->indptr;
  node->indices = node->base->indices;
  node->idtype = node->base->idtype;
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(AttachedAxisNode);

TVM_REGISTER_GLOBAL("tir.sparse.AttachedAxis").set_body_typed([](Axis base, Axis new_parent) {
  return AttachedAxis(std::move(base), std::move(new_parent));
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<AttachedAxisNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const AttachedAxisNode*>(node.get());
      p->stream << "attach(" << op->base << ", " << Downcast<Axis>(op->parent) << ")";
    });

/******** SparseBuffer ********/

PrimExpr SparseBufferNode::GetNNZ() const { return flattened->shape[0]; }

/*! \brief Default constructor of SparseBuffer */
SparseBuffer::SparseBuffer(Var data, Array<Axis> axes, DataType dtype, String name,
                           Optional<PrimExpr> extra_storage, Optional<PrimExpr> default_value,
                           Span span) {
  ObjectPtr<SparseBufferNode> node = make_object<SparseBufferNode>();
  CHECK_GT(static_cast<int>(axes.size()), 0)
      << "ValueError: A SparseBuffer should have at least one dimension";
  node->axes = axes;
  // nnz inference
  // TODO(zihao): consider flatten axis.
  PrimExpr nnz = Integer(1);
  std::unordered_map<Axis, PrimExpr, ObjectPtrHash, ObjectPtrEqual> root_nnz_map;
  std::unordered_set<Axis, ObjectPtrHash, ObjectPtrEqual> already_counted;
  for (size_t i = 0; i < axes.size(); ++i) {
    const Axis& axis = axes[i];
    const Axis& root = GetRootAxis(axis);
    if (axis->IsVariable()) {
      root_nnz_map[root] = axis->nnz;
    } else {
      root_nnz_map[root] = axis->nnz_cols.value();
    }
  }
  for (size_t i = 0; i < axes.size(); ++i) {
    const Axis& axis = axes[i];
    const Axis& root = GetRootAxis(axis);
    if (!already_counted.count(root)) {
      nnz = nnz * root_nnz_map[root];
    }
    already_counted.insert(root);
  }
  arith::Analyzer ana_;
  nnz = ana_.Simplify(nnz);

  node->data = data;
  node->extra_storage = extra_storage;
  node->name = name;
  node->dtype = dtype;
  if (!default_value) {
    node->default_value = Cast(dtype, Integer(0));
  } else {
    ICHECK(default_value.value()->dtype == dtype)
        << "sparse buffer default value should match buffer data type";
    node->default_value = default_value;
  }
  // collect shape
  Array<PrimExpr> shape;
  for (const Axis& axis : axes) {
    shape.push_back(axis->length);
  }
  node->shape = shape;
  node->strides = {};
  node->name = name;
  node->elem_offset = 0;
  node->data_alignment = runtime::kAllocAlignment;
  node->offset_factor = 1;
  node->buffer_type = BufferType::kDefault;
  // create flattened buffer
  node->flattened = Buffer(
      /*data=*/data,
      /*dtype=*/dtype,
      /*shape=*/{extra_storage.defined() ? nnz + extra_storage.value() : nnz},
      /*strides=*/{Integer(1)},
      /*elem_offset=*/PrimExpr{nullptr},
      /*name=*/name + "_data",
      /*data_alignment*/ runtime::kAllocAlignment,
      /*offset_factor=*/1,
      /*buffer_type=*/BufferType::kDefault,
      /*axis_separators=*/{},
      /*span=*/span);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(SparseBufferNode);

TVM_REGISTER_GLOBAL("tir.sparse.SparseBuffer")
    .set_body_typed([](Var data, Array<Axis> axes, DataType dtype, String name,
                       Optional<PrimExpr> extra_storage, Optional<PrimExpr> default_value,
                       Span span) {
      return SparseBuffer(std::move(data), std::move(axes), std::move(dtype), std::move(name),
                          std::move(extra_storage), std::move(default_value), std::move(span));
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<SparseBufferNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const SparseBufferNode*>(node.get());
      p->stream << "sparse_buffer(" << op->name << ", [";
      for (int i = 0, n = static_cast<int>(op->axes.size()); i < n; ++i) {
        const Axis& axis = op->axes[i];
        p->stream << axis;
        if (i < n - 1) {
          p->stream << ", ";
        }
      }
      p->stream << "]";
      if (op->extra_storage.defined()) {
        p->stream << ", " << op->extra_storage.value();
      }
      if (op->default_value.defined()) {
        p->stream << ", " << op->default_value.value();
      }
      p->stream << ")";
    });

/******** AxisKind ********/

/*! \brief Printer function of Axiskind. */
std::ostream& operator<<(std::ostream& out, AxisKind type) {
  switch (type) {
    case AxisKind::kDenseFixed:
      out << "dense-fixed";
      break;
    case AxisKind::kDenseVariable:
      out << "dense-variable";
      break;
    case AxisKind::kSparseFixed:
      out << "sparse-fixed";
      break;
    case AxisKind::kSparseVariable:
      out << "sparse-variable";
      break;
    default:
      LOG(FATAL) << "Cannot reach here";
  }
  return out;
}

/******** SpIterVar ********/

/*! \brief Default constructor of SpIterVar. */
SpIterVar::SpIterVar(Var var, bool is_reduction, Axis axis) {
  ObjectPtr<SpIterVarNode> node = make_object<SpIterVarNode>();
  node->var = Var(std::move(var));
  node->axis = std::move(axis);
  node->is_reduction = is_reduction;
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(SpIterVarNode);

TVM_REGISTER_GLOBAL("tir.sparse.SpIterVar")
    .set_body_typed([](Var var, bool is_reduction, Axis axis) {
      return SpIterVar(std::move(var), is_reduction, std::move(axis));
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<SpIterVarNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const SpIterVarNode*>(node.get());
      p->stream << "sparse_iter_var(";
      if (op->var->name_hint.length() != 0) {
        p->stream << op->var->name_hint << ", ";
      }
      p->stream << op->is_reduction << ", ";
      p->stream << op->axis << ")";
    });

/******** FormatRewriteRule ********/

/*! \brief Default constructor of FormatRewriteRule. */
FormatRewriteRule::FormatRewriteRule(String name, PrimFunc new_format_desc,
                                     Array<String> buffers_to_rewrite,
                                     Array<String> axes_before_rewrite,
                                     Array<String> axes_after_rewrite,
                                     Map<String, Array<String>> axis_map, IndexMap idx_map,
                                     IndexMap inv_idx_map) {
  ObjectPtr<FormatRewriteRuleNode> node = make_object<FormatRewriteRuleNode>();
  node->name = std::move(name);
  node->new_format_desc = std::move(new_format_desc);
  node->buffers_to_rewrite = std::move(buffers_to_rewrite);
  node->axes_before_rewrite = std::move(axes_before_rewrite);
  node->axes_after_rewrite = std::move(axes_after_rewrite);
  node->axis_map = std::move(axis_map);
  node->idx_map = std::move(idx_map);
  node->inv_idx_map = std::move(inv_idx_map);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(FormatRewriteRuleNode);

TVM_REGISTER_GLOBAL("tir.sparse.FormatRewriteRule")
    .set_body_typed([](String name, PrimFunc new_format_desc, Array<String> buffers_to_rewrite,
                       Array<String> axes_before_rewrite, Array<String> axes_after_rewrite,
                       Map<String, Array<String>> axis_map, IndexMap idx_map,
                       IndexMap inv_idx_map) {
      return FormatRewriteRule(std::move(name), std::move(new_format_desc),
                               std::move(buffers_to_rewrite), std::move(axes_before_rewrite),
                               std::move(axes_after_rewrite), std::move(axis_map),
                               std::move(idx_map), std::move(inv_idx_map));
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<FormatRewriteRuleNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const FormatRewriteRuleNode*>(node.get());
      p->stream << "sparse_format_rewrite_rule(";
      p->stream << op->name << ", ";
      p->stream << op->buffers_to_rewrite << ", ";
      p->stream << op->axes_before_rewrite << ", ";
      p->stream << op->axes_after_rewrite << ", ";
      p->stream << op->axis_map << ", ";
      p->stream << op->idx_map << ", ";
      p->stream << op->inv_idx_map;
      p->stream << ")";
    });

}  // namespace tir
}  // namespace tvm
