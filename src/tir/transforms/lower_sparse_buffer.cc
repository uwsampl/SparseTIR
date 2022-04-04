/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file lower_sparse_buffer.cc
 */

#include <tvm/arith/analyzer.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <set>
#include <utility>

#include "../../support/utils.h"
#include "../schedule/analysis.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

namespace {

/*!
 * \brief Lower the axis declaration and match_sparse_buffer to match_buffers.
 * \param f The PrimFunc whose buffer map is to be updated.
 * \return The updated buffer map.
 */
Map<Var, Buffer> UpdateBufferMap(PrimFunc f) {
  Map<Var, Buffer> buffer_map;
  for (const auto& it : f->buffer_map) {
    Var var = it.first;
    Buffer buf = it.second;
    if (const SparseBufferNode* sp_buf = buf.as<SparseBufferNode>()) {
      buffer_map.Set(var, sp_buf->flattened);
    } else {
      buffer_map.Set(var, buf);
    }
  }
  return buffer_map;
};

}  // namespace

/*!
 * \brief Lower sparse buffer access to underlying flattend buffer access by rewriting AST.
 */
class BufferTransformer : public StmtExprMutator {
 public:
  explicit BufferTransformer(Map<Var, Buffer> buffer_map) : buffer_map_(std::move(buffer_map)) {}

 private:
  /*!
   * \brief Compute the offset on underlying flattened buffer of given a given sparse buffer access.
   * \param axes The axes of the sparse buffer.
   */
  PrimExpr ComputeOffset(const Array<Axis>& axes, const Array<PrimExpr>& indices) {
    /* Algorithm
     * if fixed: index * stride
     * if variable: offset
     */
    size_t ndim = axes.size();
    arith::Analyzer ana;
    PrimExpr accum = Integer(0);
    // TODO(zihao): address the flatten axis.
    std::unordered_map<Axis, PrimExpr, StructuralHash, StructuralEqual> offset_map;
    std::unordered_map<Axis, PrimExpr, StructuralHash, StructuralEqual> root_nnz_map;
    for (size_t i = 0; i < ndim; ++i) {
      const Axis& axis = axes[i];
      const PrimExpr& index = indices[i];
      const Axis& root = GetRootAxis(axis);
      if (axis->IsVariable()) {
        if (ana.CanProveEqual(offset_map[GetParentAxis(axis)], Integer(0))) {
          // simplify expression by equality: indptr[0] = 0
          offset_map[axis] = index;
        } else {
          offset_map[axis] = index + BufferLoad(buffer_map_[axis->indptr.value()],
                                                {offset_map[GetParentAxis(axis)]});
        }
        root_nnz_map[root] = axis->nnz;
      } else {
        offset_map[axis] = index;
        root_nnz_map[root] = axis->nnz_cols.value();
      }
    }
    std::vector<bool> count(ndim, false);
    std::unordered_set<Axis, StructuralHash, StructuralEqual> visited;
    for (int i = ndim - 1; i >= 0; --i) {
      const Axis& axis = axes[i];
      const Axis& root = GetRootAxis(axis);
      count[i] = !visited.count(root);
      visited.insert(root);
    }
    for (size_t i = 0; i < ndim; ++i) {
      if (count[i]) {
        const Axis& axis = axes[i];
        const PrimExpr& index = indices[i];
        PrimExpr nnz_other_trees = Integer(1);
        PrimExpr offset_lb = offset_map[axis];
        PrimExpr offset_ub = offset_lb + 1;
        bool flattened_axis = false;
        std::unordered_set<Axis, StructuralHash, StructuralEqual> already_counted;
        for (size_t j = i + 1; j < ndim; ++j) {
          const Axis& axis_j = axes[j];
          const Axis& root = GetRootAxis(axis_j);
          if (!root.same_as(GetRootAxis(axis))) {
            if (!already_counted.count(root)) {
              nnz_other_trees = nnz_other_trees * root_nnz_map[root];
            }
            already_counted.insert(root);
          } else {
            offset_lb = BufferLoad(buffer_map_[axis_j->indptr.value()], {offset_lb});
            offset_ub = BufferLoad(buffer_map_[axis_j->indptr.value()], {offset_ub});
          }
        }
        if (flattened_axis) {
          accum = accum + index * (offset_ub - offset_lb) * nnz_other_trees;
        } else {
          accum = accum + offset_lb * nnz_other_trees;
        }
      }
    }
    return ana.Simplify(accum);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    Array<PrimExpr> indices;
    for (const PrimExpr& index : op->indices) {
      indices.push_back(VisitExpr(index));
    }
    if (const SparseBufferNode* sp_buf = op->buffer.as<SparseBufferNode>()) {
      return BufferLoad(sp_buf->flattened, {ComputeOffset(sp_buf->axes, indices)});
    } else {
      if (indices.same_as(op->indices)) {
        return GetRef<BufferLoad>(op);
      } else {
        return BufferLoad(op->buffer, indices);
      }
    }
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    PrimExpr value = VisitExpr(op->value);
    Array<PrimExpr> indices;
    for (const PrimExpr& index : op->indices) {
      indices.push_back(VisitExpr(index));
    }
    if (const SparseBufferNode* sp_buf = op->buffer.as<SparseBufferNode>()) {
      return BufferStore(sp_buf->flattened, value, {ComputeOffset(sp_buf->axes, indices)});
    } else {
      if (value.same_as(op->value) && indices.same_as(op->indices)) {
        return GetRef<BufferStore>(op);
      } else {
        return BufferStore(op->buffer, value, indices);
      }
    }
  }

  Array<BufferRegion> UpdateAccessRegion(Array<BufferRegion> regions) {
    Array<BufferRegion> new_regions;
    arith::Analyzer ana;
    for (const BufferRegion& access : regions) {
      const Buffer& buf = access->buffer;
      const Array<Range>& ranges = access->region;
      if (const SparseBufferNode* sp_buf = buf.as<SparseBufferNode>()) {
        bool single_point = true;
        Array<PrimExpr> min_indices;
        for (const Range& range : ranges) {
          min_indices.push_back(VisitExpr(range->min));
          if (!ana.CanProveEqual(range->extent, Integer(1))) {
            single_point = false;
          }
        }
        if (single_point) {
          new_regions.push_back(BufferRegion(
              sp_buf->flattened,
              {Range::FromMinExtent(ComputeOffset(sp_buf->axes, min_indices), Integer(1))}));
        } else {
          new_regions.push_back(BufferRegion(sp_buf->flattened,
                                             {Range::FromMinExtent(Integer(0), sp_buf->GetNNZ())}));
        }
      } else {
        Array<Range> new_ranges;
        for (const Range& range : ranges) {
          PrimExpr new_min = VisitExpr(range->min);
          PrimExpr new_extent = VisitExpr(range->extent);
          if (new_min.same_as(range->min) && new_extent.same_as(range->extent)) {
            new_ranges.push_back(range);
          } else {
            new_ranges.push_back(Range::FromMinExtent(new_min, new_extent));
          }
        }
        if (new_ranges.same_as(ranges)) {
          new_regions.push_back(access);
        } else {
          new_regions.push_back(BufferRegion(buf, new_ranges));
        }
      }
    }
    return new_regions;
  }

  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    // if some iter vars are binded to 0, substitute corresponding variable with 0
    Array<PrimExpr> new_iter_values;
    for (const PrimExpr& iter_value: op->iter_values) {
      new_iter_values.push_back(VisitExpr(iter_value));
    }
    arith::Analyzer ana;
    Block block = op->block;
    Optional<Stmt> init = block->init;
    Stmt body = block->body;
    std::unordered_map<const VarNode*, PrimExpr> var_map_;
    for (size_t i = 0; i < new_iter_values.size(); ++i) {
      if (ana.CanProveEqual(new_iter_values[i], Integer(0))) {
        var_map_[block->iter_vars[i]->var.get()] = Integer(0);
      }
    }
    if (init.defined()) {
      init = Substitute(init.value(), var_map_);
    }
    body = Substitute(body, var_map_);
    auto block_node = CopyOnWrite(block.get());
    block_node->init = init;
    block_node->body = body;
    Stmt new_block = VisitStmt(Block(block_node));
    auto block_realize_node = CopyOnWrite(op);
    block_realize_node->iter_values = new_iter_values;
    block_realize_node->predicate = VisitExpr(op->predicate);
    block_realize_node->block = Downcast<Block>(new_block);
    return BlockRealize(block_realize_node);
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    // visit body and init block recursively.
    Optional<Stmt> init = NullOpt;
    if (op->init.defined()) {
      init = VisitStmt(op->init.value());
    }
    Stmt body = VisitStmt(op->body);
    auto n = CopyOnWrite(op);
    n->iter_vars = std::move(op->iter_vars);
    // update alloc_buffer
    Array<Buffer> new_alloc_buffers;
    for (const Buffer& buf : n->alloc_buffers) {
      if (const SparseBufferNode* sp_buf = buf.as<SparseBufferNode>()) {
        new_alloc_buffers.push_back(sp_buf->flattened);
      } else {
        new_alloc_buffers.push_back(buf);
      }
    }
    n->alloc_buffers = std::move(new_alloc_buffers);
    // TODO(zihao): update match buffer
    // update read/write regions in lower buffer process.
    n->reads = UpdateAccessRegion(std::move(n->reads));
    n->writes = UpdateAccessRegion(std::move(n->writes));
    n->body = std::move(body);
    n->init = std::move(init);
    return Block(n);
  }
  Map<Var, Buffer> buffer_map_;
};

PrimFunc LowerSparseBuffer(PrimFunc f) {
  // Only apply this pass to TIR that is not from TE schedules
  if (!IsFromLegacyTESchedule(f) && SparseTIRLevel(f) == 1) {
    PrimFuncNode* fptr = f.CopyOnWrite();
    // Step 1. Update the PrimFunc's buffer map.
    fptr->buffer_map = std::move(UpdateBufferMap(f));
    // Step 2. Remove sp_axes and sp_axes_param
    fptr->sp_axes.clear();
    // Step 3. Lower sparse buffers.
    fptr->body = BufferTransformer(fptr->buffer_map)(std::move(fptr->body));
    // Step 4. Lower sparse tir level
    Map<String, ObjectRef> new_attr_dict = fptr->attrs->dict;
    new_attr_dict.Set("sparse_tir_level", Integer(0));
    fptr->attrs = DictAttrs(new_attr_dict);
    return f;
  } else {
    return f;
  }
}

namespace transform {

/*!
 * \brief The lowering pass from TIR to Sparse TIR.
 */
Pass LowerSparseBuffer() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return LowerSparseBuffer(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerSparseBuffer", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerSparseBuffer").set_body_typed(LowerSparseBuffer);

}  // namespace transform

}  // namespace tir
}  // namespace tvm