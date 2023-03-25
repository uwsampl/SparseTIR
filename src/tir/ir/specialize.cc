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
 * \file src/tir/ir/specialize.cc
 * \brief Specialize parameters of PrimFunc.
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <functional>

#include "functor_common.h"

namespace tvm {
namespace tir {

using VarMap = std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual>;

/**************** Helper functions ****************/

/*! \brief Helper function to check whether the given var is in function parameter list. */
inline bool IsParam(const PrimFunc& func, const Var& param) {
  return std::any_of(func->params.begin(), func->params.end(),
                     [&](const Var& var) { return var.same_as(param); });
}

/**************** Specializer ****************/

// Try fold constants if op's child get specialized to constant.
#define DEFINE_SPECIALIZER_BINARY_OP_MUTATE(BinaryNode, BinaryFunc) \
  PrimExpr VisitExpr_(const BinaryNode* op) final {                 \
    PrimExpr a = VisitExpr(op->a);                                  \
    PrimExpr b = VisitExpr(op->b);                                  \
    if (a.same_as(op->a) && b.same_as(op->b)) {                     \
      return GetRef<PrimExpr>(op);                                  \
    } else {                                                        \
      return BinaryFunc(a, b);                                      \
    }                                                               \
  }
#define DEFINE_SPECIALIZER_UNARY_OP_MUTATE(UnaryNode, UnaryFunc) \
  PrimExpr VisitExpr_(const UnaryNode* op) final {               \
    PrimExpr a = VisitExpr(op->a);                               \
    if (a.same_as(op->a)) {                                      \
      return GetRef<PrimExpr>(op);                               \
    } else {                                                     \
      return UnaryFunc(a);                                       \
    }                                                            \
  }

/*! \brief Mutator to specialize function and remove const parameters */
class PrimFuncSpecializer : public StmtExprMutator {
 public:
  explicit PrimFuncSpecializer(const VarMap& var_map) : var_map_(var_map) {}

  static PrimFunc Specialize(PrimFunc f, const VarMap& var_map) {
    PrimFuncSpecializer specializer(var_map);

    // Update sp_axes
    Array<Axis> sp_axes;
    bool sp_axes_updated = false;
    for (const Axis& axis : f->sp_axes) {
      Axis new_axis = specializer.MutateAxis(axis);
      sp_axes.push_back(new_axis);
      if (!new_axis.same_as(axis)) {
        sp_axes_updated = true;
        specializer.axis_map_[axis] = new_axis;
      }
    }

    // Updating Buffer map
    Map<Var, Buffer> buffer_map;
    bool buffer_map_updated = false;
    for (const auto& it : f->buffer_map) {
      const Var& var = it.first;
      const Buffer& buffer = it.second;
      Buffer new_buffer = specializer.MutateBuffer(buffer);
      buffer_map.Set(var, new_buffer);
      if (!new_buffer.same_as(buffer)) {
        buffer_map_updated = true;
        specializer.buffer_map_[buffer] = new_buffer;
      }
    }

    // Updating parmeters
    Array<Var> params;
    bool param_updated = false;
    for (const auto& var : f->params) {
      // Remove parmeters which has been specialized.
      if (var_map.find(var) == var_map.end()) {
        params.push_back(var);
      } else {
        param_updated = true;
      }
    }

    // Updating function body
    Stmt body = specializer(f->body);

    if (param_updated || buffer_map_updated || sp_axes_updated || !f->body.same_as(body)) {
      PrimFuncNode* f_ptr = f.CopyOnWrite();
      f_ptr->params = std::move(params);
      f_ptr->buffer_map = std::move(buffer_map);
      f_ptr->body = std::move(body);
      f_ptr->sp_axes = std::move(sp_axes);
    }
    return f;
  }

 private:
  Stmt VisitStmt_(const BlockNode* op) final {
    // Step.0. Define buffer mappings which is allocated inside the block
    Array<Buffer> alloc_buffers = MutateArray(
        op->alloc_buffers,
        std::bind(&PrimFuncSpecializer::MutateAllocBuffer, this, std::placeholders::_1));

    // Step.1. Recursively visit block body
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BlockNode>();
    ICHECK(op != nullptr);

    Array<BufferDomain> buf_doms = MutateArray(
        op->buf_doms,
        std::bind(&PrimFuncSpecializer::MutateBufferDomain, this, std::placeholders::_1));
    Array<BufferRegion> reads = MutateArray(
        op->reads,
        std::bind(&PrimFuncSpecializer::MutateBufferRegion, this, std::placeholders::_1));
    Array<BufferRegion> writes = MutateArray(
        op->writes,
        std::bind(&PrimFuncSpecializer::MutateBufferRegion, this, std::placeholders::_1));
    Array<IterVar> iter_vars = MutateArray(
        op->iter_vars, std::bind(&PrimFuncSpecializer::MutateIterVar, this, std::placeholders::_1));

    if (alloc_buffers.same_as(op->alloc_buffers) && reads.same_as(op->reads) &&
        writes.same_as(op->writes) && buf_doms.same_as(op->buf_doms) &&
        iter_vars.same_as(op->iter_vars)) {
      return GetRef<Block>(op);
    } else {
      ObjectPtr<BlockNode> n = CopyOnWrite(op);
      n->alloc_buffers = std::move(alloc_buffers);
      n->buf_doms = std::move(buf_doms);
      n->reads = std::move(reads);
      n->writes = std::move(writes);
      n->iter_vars = std::move(iter_vars);
      return Stmt(n);
    }
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BufferStoreNode>();
    ICHECK(op != nullptr);
    auto it = buffer_map_.find(op->buffer);
    if (it == buffer_map_.end()) {
      return GetRef<BufferStore>(op);
    } else {
      auto n = CopyOnWrite(op);
      n->buffer = it->second;
      return Stmt(n);
    }
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<BufferLoadNode>();
    ICHECK(op != nullptr);
    auto it = buffer_map_.find(op->buffer);
    if (it == buffer_map_.end()) {
      return GetRef<BufferLoad>(op);
    } else {
      auto n = make_object<BufferLoadNode>(*op);
      n->buffer = it->second;
      return PrimExpr(n);
    }
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    auto it = var_map_.find(GetRef<Var>(op));
    if (it == var_map_.end()) {
      return GetRef<PrimExpr>(op);
    } else {
      return it->second;
    }
  }

  Stmt VisitStmt_(const SparseIterationNode* op) {
    Optional<Stmt> init = NullOpt;
    if (op->init.defined()) {
      init = VisitStmt(op->init.value());
    }
    Stmt body = VisitStmt(op->body);
    Array<SpIterVar> sp_iter_vars =
        MutateArray(op->sp_iter_vars,
                    std::bind(&PrimFuncSpecializer::MutateSpIterVar, this, std::placeholders::_1));

    if (init.same_as(op->init) && body.same_as(op->body) &&
        sp_iter_vars.same_as(op->sp_iter_vars)) {
      return GetRef<SparseIteration>(op);
    } else {
      auto n = CopyOnWrite(op);
      n->init = std::move(init);
      n->body = std::move(body);
      n->sp_iter_vars = std::move(sp_iter_vars);
      return Stmt(n);
    }
  }

  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(AddNode, add);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(SubNode, sub);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(MulNode, mul);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(DivNode, div);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(ModNode, truncmod);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(FloorDivNode, floordiv);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(FloorModNode, floormod);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(MaxNode, max);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(MinNode, min);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(EQNode, equal);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(NENode, not_equal);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(LTNode, less);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(LENode, less_equal);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(GTNode, greater);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(GENode, greater_equal);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(AndNode, logical_and);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(OrNode, logical_or);
  DEFINE_SPECIALIZER_UNARY_OP_MUTATE(NotNode, logical_not);

 private:
  Buffer MutateBuffer(const Buffer& buffer) {
    if (const SparseBufferNode* sp_buf = buffer.as<SparseBufferNode>()) {
      Array<Axis> axes = MutateArray(
          sp_buf->axes, std::bind(&PrimFuncSpecializer::MutateAxis, this, std::placeholders::_1));
      if (axes.same_as(sp_buf->axes)) {
        return buffer;
      } else {
        return SparseBuffer(sp_buf->data, std::move(axes), sp_buf->dtype, sp_buf->name,
                            sp_buf->extra_storage, sp_buf->default_value, sp_buf->span);
      }
    } else {
      Array<PrimExpr> shape =
          MutateArray(buffer->shape, [this](const PrimExpr& e) { return VisitExpr(e); });
      Array<PrimExpr> strides =
          MutateArray(buffer->strides, [this](const PrimExpr& e) { return VisitExpr(e); });

      PrimExpr elem_offset = VisitExpr(buffer->elem_offset);

      if (buffer->elem_offset.same_as(elem_offset) && buffer->shape.same_as(shape) &&
          buffer->strides.same_as(strides)) {
        return buffer;
      } else {
        auto n = make_object<BufferNode>(*buffer.get());
        n->elem_offset = std::move(elem_offset);
        n->shape = std::move(shape);
        n->strides = std::move(strides);
        return Buffer(n);
      }
    }
  }

  Axis MutateAxis(const Axis& axis) {
    // NOTE(zihao): change this if there are new kind of axis.
    // Order: from inherited class to base class.
    if (axis_map_.find(axis) != axis_map_.end()) {
      return axis_map_[axis];
    } else {
      if (const FlattenedAxisNode* flattened_axis = axis.as<FlattenedAxisNode>()) {
        Array<Axis> axes = MutateArray(flattened_axis->axes,
                                       [this](const Axis& axis) { return MutateAxis(axis); });
        PrimExpr flattened_nnz = VisitExpr(flattened_axis->flattened_nnz);
        Buffer offset = MutateBuffer(flattened_axis->offset);
        if (axes.same_as(flattened_axis->axes) &&
            flattened_nnz.same_as(flattened_axis->flattened_nnz) &&
            offset.same_as(flattened_axis->offset)) {
          return axis;
        } else {
          return FlattenedAxis(axis->name, axes, flattened_nnz, offset);
        }
      } else if (const AttachedAxisNode* attached_axis = axis.as<AttachedAxisNode>()) {
        Axis base = MutateAxis(attached_axis->base);
        Axis new_parent = MutateAxis(GetParentAxis(axis));
        if (base.same_as(attached_axis->base) && new_parent.same_as(GetParentAxis(axis))) {
          return axis;
        } else {
          return AttachedAxis(base, new_parent);
        }
      } else if (const FusedAxisNode* fused_axis = axis.as<FusedAxisNode>()) {
        Array<Axis> group =
            MutateArray(fused_axis->group,
                        std::bind(&PrimFuncSpecializer::MutateAxis, this, std::placeholders::_1));
        if (group.same_as(fused_axis->group)) {
          return axis;
        } else {
          return FusedAxis(group, fused_axis->index);
        }
      } else {
        switch (axis->kind()) {
          case AxisKind::kDenseFixed: {
            PrimExpr length = VisitExpr(axis->length);
            if (length.same_as(axis->length)) {
              return axis;
            } else {
              return Axis(axis->name, NullOpt, length, length, length, NullOpt, NullOpt,
                          axis->idtype);
            }
            break;
          }
          case AxisKind::kDenseVariable: {
            Axis old_parent = GetParentAxis(axis);
            Axis parent = MutateAxis(old_parent);
            PrimExpr length = VisitExpr(axis->length);
            PrimExpr nnz = VisitExpr(axis->nnz);
            if (parent.same_as(old_parent) && length.same_as(axis->length) &&
                nnz.same_as(axis->nnz)) {
              return axis;
            } else {
              return Axis(axis->name, parent, length, nnz, NullOpt, axis->indptr, NullOpt,
                          axis->idtype);
            }
            break;
          }
          case AxisKind::kSparseFixed: {
            Axis old_parent = GetParentAxis(axis);
            Axis parent = MutateAxis(old_parent);
            PrimExpr length = VisitExpr(axis->length);
            PrimExpr nnz = VisitExpr(axis->nnz);
            PrimExpr nnz_cols = VisitExpr(axis->nnz_cols.value());
            if (parent.same_as(old_parent) && length.same_as(axis->length) &&
                nnz_cols.same_as(axis->nnz_cols.value())) {
              return axis;
            } else {
              return Axis(axis->name, parent, length, nnz, nnz_cols, NullOpt, axis->indices,
                          axis->idtype);
            }
            break;
          }
          case AxisKind::kSparseVariable: {
            Axis old_parent = GetParentAxis(axis);
            Axis parent = MutateAxis(old_parent);
            PrimExpr length = VisitExpr(axis->length);
            PrimExpr nnz = VisitExpr(axis->nnz);
            if (parent.same_as(old_parent) && length.same_as(axis->length)) {
              return axis;
            } else {
              return Axis(axis->name, parent, length, nnz, NullOpt, axis->indptr, axis->indices,
                          axis->idtype);
            }
            break;
          }
          default:
            throw;
        }
      }
    }
  }

  IterVar MutateIterVar(const IterVar& iter_var) { return iter_var; }

  SpIterVar MutateSpIterVar(const SpIterVar& sp_iter_var) {
    Axis axis = MutateAxis(sp_iter_var->axis);

    if (axis.same_as(sp_iter_var->axis)) {
      return sp_iter_var;
    } else {
      return SpIterVar(sp_iter_var->var, sp_iter_var->is_reduction, std::move(axis));
    }
  }

  Range MutateRange(const Range& range) {
    PrimExpr min = this->VisitExpr(range->min);
    PrimExpr extent = this->VisitExpr(range->extent);
    if (min.same_as(range->min) && extent.same_as(range->extent)) {
      return range;
    } else {
      return Range::FromMinExtent(std::move(min), std::move(extent));
    }
  }

  Buffer MutateAllocBuffer(const Buffer& alloc_buf) {
    Buffer buf = MutateBuffer(alloc_buf);
    if (buf.same_as(alloc_buf)) {
      return alloc_buf;
    } else {
      ICHECK(buffer_map_.find(alloc_buf) == buffer_map_.end());
      buffer_map_[alloc_buf] = buf;
      return buf;
    }
  }

  BufferRegion MutateBufferRegion(const BufferRegion& buffer_region) {
    auto it = buffer_map_.find(buffer_region->buffer);
    Array<Range> region =
        MutateArray(buffer_region->region,
                    std::bind(&PrimFuncSpecializer::MutateRange, this, std::placeholders::_1));
    if (it == buffer_map_.end() && region.same_as(buffer_region->region)) {
      return buffer_region;
    } else {
      return BufferRegion(it->second, std::move(region));
    }
  }

  BufferDomain MutateBufferDomain(const BufferDomain& buf_dom) {
    auto it = buffer_map_.find(buf_dom->buffer);
    Range dom = MutateRange(buf_dom->dom);
    if (it == buffer_map_.end() && dom.same_as(buf_dom->dom)) {
      return buf_dom;
    } else {
      return BufferDomain(it->second, std::move(dom));
    }
  }

 private:
  /*! \brief The vars to be substitute and their values */
  const VarMap& var_map_;
  /*! \brief map from old buffer to mutated buffer */
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_map_;
  /*! \brief map from old axis to mutated axis */
  std::unordered_map<Axis, Axis, ObjectPtrHash, ObjectPtrEqual> axis_map_;
};

/*!
 * \brief Update Specialize var map with buffer matching.
 * \param func The function to be specialized.
 * \param param The given function parameter
 * \param specific_buf The matching buffer.
 * \param var_map The var mapping to be updated.
 * \note This function will match target buffer's shape, strides and element_offset
 *   For example, we define a buffer in PrimFunc:
 *   A = T.match_buffer(a, [m, n])
 *
 *   Then we match it with a buffer B =  tir.decl_buffer((8, 16))
 *
 *   It means we have two var mappings here: m = 8 and n = 16
 *
 *   If the buffer signature is not a Var, the mapping will fail.
 *   e.g. A = T.match_buffer(a, [m * 2, n + 1])
 */
void UpdateSpecializeVarMap(const PrimFunc& func, const Var& param, const Buffer& specific_buf,
                            VarMap* var_map) {
  // preliminaries
  tir::ExprDeepEqual equal;

  auto it = func->buffer_map.find(param);
  CHECK(it != func->buffer_map.end())
      << "ValueError: specialize expects param to be in PrimFunc's buffer_map";
  const Buffer& buf_to_specialize = (*it).second;

  // build var mapping using specific_buf's parameters
  auto build_var_mapping = [&](const PrimExpr& new_expr, const PrimExpr& old_expr) {
    if (!equal(new_expr, old_expr)) {
      CHECK(old_expr->IsInstance<VarNode>())
          << "TypeError: The signature of target buffer exprected an independent Var, but got "
          << old_expr << ".";
      const Var& var = Downcast<Var>(old_expr);
      auto it = var_map->find(var);
      if (it != var_map->end()) {
        CHECK(equal(it->second, new_expr))
            << "ValueError: The assigned value of var " << var << " mismatched. " << it->second
            << " vs. " << new_expr << ".";
      } else {
        (*var_map)[var] = new_expr;
      }
    }
  };

  // Check buffer dimensions
  CHECK(specific_buf->shape.size() == buf_to_specialize->shape.size())
      << "ValueError: The buffer dimensions mismatched" << buf_to_specialize->shape.size()
      << " vs. " << specific_buf->shape.size() << ".";

  CHECK(specific_buf->strides.size() == buf_to_specialize->strides.size())
      << "ValueError: The buffer strides dimensions mismatched" << buf_to_specialize->strides.size()
      << " vs. " << specific_buf->strides.size() << ".";

  // Updating var mapping using specific_expr
  for (size_t i = 0; i < specific_buf->shape.size(); ++i) {
    build_var_mapping(specific_buf->shape[i], buf_to_specialize->shape[i]);
  }
  for (size_t i = 0; i < specific_buf->strides.size(); ++i) {
    build_var_mapping(specific_buf->strides[i], buf_to_specialize->strides[i]);
  }
  build_var_mapping(specific_buf->elem_offset, buf_to_specialize->elem_offset);

  // Check data_alignment and offset_factor.
  // These two signatures are int, so we do not need map them.
  CHECK_EQ(specific_buf->data_alignment, buf_to_specialize->data_alignment)
      << "ValueError: The buffer data_alignment mismatched" << buf_to_specialize->data_alignment
      << " vs. " << specific_buf->data_alignment << ".";

  CHECK_EQ(specific_buf->offset_factor, buf_to_specialize->offset_factor)
      << "ValueError: The buffer offset_factor mismatched" << buf_to_specialize->offset_factor
      << " vs. " << specific_buf->offset_factor << ".";
}

/*!
 * \brief Update Specialize var map with parameter value.
 * \param func The function to be specialized.
 * \param param The given function parameter
 * \param specific_expr The parameter value.
 * \param var_map The var mapping to be updated.
 */
void UpdateSpecializeVarMap(const PrimFunc& func, const Var& param, const PrimExpr& specific_expr,
                            VarMap* var_map) {
  // check param is in PrimFunc's parameters
  CHECK(IsParam(func, param)) << "ValueError: Specialize expects param to be in PrimFunc's params";
  // specialize a param not in buffer_map
  CHECK_EQ(func->buffer_map.count(param), 0)
      << "ValueError: Specialize expects param to not be in PrimFunc's buffer_map";
  // build var mapping using specific_expr
  (*var_map)[param] = specific_expr;
}

/**************** Implementation ****************/

PrimFunc Specialize(PrimFunc func, const Map<Var, ObjectRef>& param_map) {
  VarMap var_map;
  for (const auto& kv : param_map) {
    const Var& param = kv.first;
    const ObjectRef& instance = kv.second;
    if (instance->IsInstance<BufferNode>()) {
      UpdateSpecializeVarMap(func, param, Downcast<Buffer>(instance), &var_map);
    } else if (instance->IsInstance<PrimExprNode>()) {
      UpdateSpecializeVarMap(func, param, Downcast<PrimExpr>(instance), &var_map);
    } else {
      LOG(FATAL) << "TypeError: specialize expected instance to be Buffer or PrimExpr, but got "
                 << instance->GetTypeKey();
    }
  }
  return PrimFuncSpecializer::Specialize(func, std::move(var_map));
}

/**************** FFI ****************/

TVM_REGISTER_GLOBAL("tir.Specialize").set_body_typed(Specialize);

}  // namespace tir
}  // namespace tvm
