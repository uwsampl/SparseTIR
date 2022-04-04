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
 * \file lower_sparse_iter.cc
 */

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
 * \brief Collect the ancestors of the given axis along the path to the root.
 * \note self not included.
 */
Array<Axis> CollectAncestors(Axis axis, int max_depth = -1) {
  Array<Axis> parents, ret;
  Optional<ObjectRef> parent = axis->parent;
  while (parent.defined()) {
    parents.push_back(Downcast<Axis>(parent.value()));
    axis = parents.back();
    parent = axis->parent;
    if (max_depth >= 0) {
      if (int(parents.size()) >= max_depth) {
        break;
      }
    }
  }
  for (int i = int(parents.size()) - 1; i >= 0; i--) {
    ret.push_back(parents[i]);
  }
  return ret;
}

/*!
 * \brief Add indptr and indices buffer-matches to PrimFunc's buffer map.
 */
std::tuple<Map<Axis, Buffer>, Map<Axis, Buffer>, Map<Var, Buffer>, Array<Axis>> UpdateBufferMap(
    PrimFunc f) {
  Map<Var, Buffer> buffer_map = f->buffer_map;
  Map<Axis, Buffer> axis_indptr_map;
  Map<Axis, Buffer> axis_indices_map;
  std::unordered_map<const AxisNode*, Axis> to_dense_map;
  Array<Axis> new_sp_axes;
  for (const Axis& axis : f->sp_axes) {
    new_sp_axes.push_back(axis);
    // TODO(Zihao): special handle of FlattenedAxis
    if (axis->IsVariable()) {
      // axis is variable, generate the indptr sparse buffers.
      String indptr_name = axis->name + "_indptr";
      Var indptr(indptr_name + ".data", PointerType(PrimType(axis->idtype), "global"));
      Array<Axis> ancestors = CollectAncestors(axis);
      Axis parent = ancestors.back();
      if (parent->IsSparse()) {
        if (!to_dense_map.count(parent.get())) {
          new_sp_axes.push_back(ToDenseAxis(parent));
          to_dense_map[parent.get()] = new_sp_axes.back();
        }
        ancestors.Set(ancestors.size() - 1, to_dense_map[parent.get()]);
      }
      SparseBuffer sp_buf(indptr, ancestors, axis->idtype, indptr_name, Integer(1));
      buffer_map.Set(axis->indptr.value(), sp_buf);
      axis_indptr_map.Set(axis, sp_buf);
    }
    if (axis->IsSparse()) {
      // axis is sparse, generate the indices sparse buffers.
      String indices_name = axis->name + "_indices";
      Var indices(indices_name + ".data", PointerType(PrimType(axis->idtype), "global"));
      Array<Axis> ancestors = CollectAncestors(axis);
      if (!to_dense_map.count(axis.get())) {
        new_sp_axes.push_back(ToDenseAxis(axis));
        to_dense_map[axis.get()] = new_sp_axes.back();
      }
      ancestors.push_back(to_dense_map[axis.get()]);
      SparseBuffer sp_buf(indices, ancestors, axis->idtype, indices_name, NullOpt);
      buffer_map.Set(axis->indices.value(), sp_buf);
      axis_indices_map.Set(axis, sp_buf);
    }
  }
  return {axis_indptr_map, axis_indices_map, buffer_map, new_sp_axes};
}

/*!
 * \brief Create an intermediate buffer with specified name and data type
 * \param name The specified name
 * \param dtype The specified data type
 * \return The created buffer
 */
Buffer MakeScratchpad(String name, const DataType& dtype) {
  return Buffer(/*ptr=*/Var(name, PointerType(PrimType(dtype), "local")),
                /*dtype=*/dtype,
                /*shape=*/{Integer(1)},
                /*strides=*/{Integer(1)},
                /*elem_offset=*/PrimExpr{nullptr},
                /*name=*/name,
                /*data_alignment=*/0,
                /*offset_factor=*/0,
                /*buffer_type=*/kDefault);
}

Axis GetAxisBeforeFuse(const Axis& axis) {
  if (const FusedAxisNode* fused_axis = axis.as<FusedAxisNode>()) {
    return fused_axis->group[fused_axis->index];
  } else {
    return axis;
  }
}

}  // namespace

/*!
 * \brief Auxlliary context data structure for lower sparse iter vars.
 */
class LowerSparseIterContext {
 public:
  /*! \brief Enter a new scope. */
  void EnterScope(const std::unordered_map<const VarNode*, arith::IntSet>& base_dom_map) {
    stack_.push_back(Info());
    top()->dom_map_ = base_dom_map;
  }

  /*! \brief Exit the current scope. */
  void ExitScope() {
    /* aggregate dom map. */
    for (auto it : top()->dom_map_) {
      if (stack_.size() > 1) {
        stack_[stack_.size() - 2].dom_map_.insert(it);
      }
    }
    stack_.pop_back();
  }

  /*! \brief Change the is_collecting_regions flag. */
  void CollectRegion(bool is_collecting_regions) {
    top()->is_collecting_regions = is_collecting_regions;
  }

  /*! \brief Whether the visitor is collecting read/write regions or not. */
  bool IsCollectingRegions() { return top()->is_collecting_regions; }

  /*! \brief Update read regions. */
  void UpdateRead(Buffer buffer, const std::vector<arith::IntSet>& region) {
    Update(top()->read_buffers_, top()->read_regions_, buffer, region);
  }

  /*! \brief Update write regions. */
  void UpdateWrite(Buffer buffer, const std::vector<arith::IntSet>& region) {
    Update(top()->write_buffers, top()->write_regions_, buffer, region);
  }

  /*! \brief Return the collected read regions. */
  Array<BufferRegion> CollectReadRegions() {
    return std::move(CollectRegions(top()->read_buffers_, top()->read_regions_));
  }

  /*! \brief Return the collected write regions. */
  Array<BufferRegion> CollectWriteRegions() {
    return std::move(CollectRegions(top()->write_buffers, top()->write_regions_));
  }

  /*! \brief Update the variable-domain map. */
  void AddVarDom(Var var, arith::IntSet dom) { top()->dom_map_[var.get()] = dom; }

  /*! \brief Get the variable-domain map. */
  std::unordered_map<const VarNode*, arith::IntSet>& GetDomMap() { return top()->dom_map_; };

  /*! \brief Add an axis-itervar mapping to the context. */
  void AddAxisIterVar(Axis axis, SpIterVar iter_var) {
    top()->axis_itervar_map_[axis.get()] = iter_var;
  }

  /*! \brief Return the itervar corresponding to an axis. */
  Optional<SpIterVar> GetIterVarFromAxis(Axis axis) {
    for (int i = stack_.size() - 1; i >= 0; i--) {
      auto it = stack_[i].axis_itervar_map_.find(axis.get());
      if (it != stack_[i].axis_itervar_map_.end()) {
        return it->second;
      }
    }
    return NullOpt;
  }

  /*! \brief Add an var-itervar mapping to the context. */
  void AddVarIterVar(Var var, SpIterVar iter_var) { top()->var_itervar_map_[var.get()] = iter_var; }

  /*! \brief Return the sparse itervar corresponding to a variable. */
  Optional<SpIterVar> GetSpIterVarFromVar(Var var) {
    for (int i = stack_.size() - 1; i >= 0; i--) {
      auto it = stack_[i].var_itervar_map_.find(var.get());
      if (it != stack_[i].var_itervar_map_.end()) {
        return it->second;
      }
    }
    return NullOpt;
  }

  /*! \brief Add emitted binary-search blocks, and corresonding buffer/range to the context. */
  void AddBinarySearch(BlockRealize blk, const Buffer& buf, const Range& range) {
    top()->bsearch_blks.push_back(std::move(blk));
    top()->bsearch_ranges.push_back({buf, Var(buf->name, buf->dtype), range});
  }

  /*! \brief Return the emitted blocks for binary search in the context. */
  std::vector<BlockRealize>& GetBinarySearchBlocks() const { return top()->bsearch_blks; }

  /*! \brief Return the buffer-variable-ranges for binary search in the context. */
  std::vector<std::tuple<Buffer, Var, Range>>& GetBinarySearchRanges() const {
    return top()->bsearch_ranges;
  }

  /*!
   * \brief Clear read/write buffers and regions in the context.
   */
  void ClearReadWriteBufferRegions() {
    top()->read_buffers_.clear();
    top()->read_regions_.clear();
    top()->write_buffers.clear();
    top()->write_regions_.clear();
  }

 private:
  /*! \brief Data structures storing sparse-iteration local information. */
  struct Info {
    bool is_collecting_regions = false;
    /*! \brief Iteration range for loop_vars */
    std::unordered_map<const VarNode*, arith::IntSet> dom_map_;
    /*! \brief The buffers that the current block reads */
    std::vector<Buffer> read_buffers_;
    /*! \brief The buffers that the current block writes */
    std::vector<Buffer> write_buffers;
    /*! \brief The opaque buffer which is access by buffer.data */
    std::vector<std::vector<tvm::arith::IntSet>> read_regions_;
    /*! \brief The write regions of the current block */
    std::vector<std::vector<tvm::arith::IntSet>> write_regions_;
    /*! \brief The map from axis to corresponding sparse iter var. */
    std::unordered_map<const AxisNode*, SpIterVar> axis_itervar_map_;
    /*! \brief The map from Var to corresponding Sparse IterVar. */
    std::unordered_map<const VarNode*, SpIterVar> var_itervar_map_;
    /*! \brief Allocated blocks for binary search. */
    std::vector<BlockRealize> bsearch_blks;
    /*! \brief Buffer-variable-ranges for binary search. */
    std::vector<std::tuple<Buffer, Var, Range>> bsearch_ranges;
  };

  /*! \brief Get the information corresponding to the top sparse-iteration in the stack. */
  inline Info* top() const { return const_cast<Info*>(&stack_.back()); }

  /*! \brief Update buffer regions. */
  void Update(std::vector<Buffer>& buffers, std::vector<std::vector<arith::IntSet>>& regions,
              Buffer buffer, const std::vector<arith::IntSet>& region) {
    ICHECK_EQ(buffers.size(), regions.size())
        << " Expected the buffer and regions to have the same size ";
    for (size_t i = 0; i < regions.size(); ++i) {
      if (buffers[i].same_as(buffer)) {
        ICHECK_EQ(regions[i].size(), region.size()) << "Inconsistent buffer dimension";
        for (size_t j = 0; j < region.size(); ++j) {
          regions[i][j] = arith::Union({regions[i][j], region[j]});
        }
        return;
      }
    }
    buffers.push_back(std::move(buffer));
    regions.push_back(std::move(region));
  }

  /*! \brief Return the array of BufferRegion's given buffer array and region array. */
  Array<BufferRegion> CollectRegions(const std::vector<Buffer>& buffers,
                                     const std::vector<std::vector<tvm::arith::IntSet>>& regions) {
    ICHECK_EQ(buffers.size(), regions.size());
    Array<BufferRegion> res;
    res.reserve(buffers.size());
    for (size_t i = 0; i < regions.size(); ++i) {
      Array<Range> region;
      region.reserve(regions[i].size());
      ICHECK_EQ(buffers[i]->shape.size(), regions[i].size());
      for (size_t j = 0; j < regions[i].size(); j++) {
        const tvm::arith::IntSet& range = regions[i][j];
        region.push_back(range.CoverRange(Range::FromMinExtent(0, buffers[i]->shape[j])));
      }
      res.push_back(BufferRegion(buffers[i], region));
    }
    return res;
  }

  /*! The stack of sparse-iteration local informations. */
  std::vector<Info> stack_;
};

/*!
 * \brief Rewrite buffer load to variable, for buffer region analysis regarding binary search
 * results.
 */
class BufVarRewriter : public ExprMutator {
 public:
  explicit BufVarRewriter(LowerSparseIterContext* ctx) {
    std::vector<std::tuple<Buffer, Var, Range>>& bsearch_ranges = ctx->GetBinarySearchRanges();
    for (auto& it : bsearch_ranges) {
      Buffer buf;
      Var var;
      Range range;
      std::tie(buf, var, range) = it;
      buf_var_map_[buf.get()] = var;
    }
  }

 private:
  PrimExpr VisitExpr_(const BufferLoadNode* op) {
    if (buf_var_map_.count(op->buffer.get())) {
      return buf_var_map_[op->buffer.get()];
    }
    return GetRef<BufferLoad>(op);
  }
  std::unordered_map<const BufferNode*, Var> buf_var_map_;
};

/*!
 * \brief Lower sparse iterations by rewriting AST.
 */
class IterTransformer : public StmtExprMutator {
 public:
  explicit IterTransformer(Map<Axis, Buffer> axis_indptr_map, Map<Axis, Buffer> axis_indices_map,
                           const Array<Axis>& sp_axes)
      : axis_indptr_map_(std::move(axis_indptr_map)),
        axis_indices_map_(std::move(axis_indices_map)),
        bsearch_blk_counter(0) {
    CreateBaseDomMap(sp_axes);
  }

  Array<Buffer> root_alloc_buffers;  // allocated buffers in root block.
  Array<Stmt> root_stmts;            // initialization statements in root block.
  Optional<Buffer> bsearch_low;      // local buffer `low` for binary search.
  Optional<Buffer> bsearch_high;     // local buffer `high` for binary search.
 private:
  /*! \brief Create base dom map: each axis parameters should be greater than 0. */
  void CreateBaseDomMap(const Array<Axis>& axes) {
    for (const Axis& axis : axes) {
      const VarNode* var_length = axis->length.as<VarNode>();
      if (var_length) {
        if (!base_dom_map_.count(var_length)) {
          base_dom_map_[var_length] = arith::IntSet::FromMinExtent(Integer(1), axis->length);
        }
      }
      const VarNode* var_nnz = axis->nnz.as<VarNode>();
      if (var_nnz) {
        if (!base_dom_map_.count(var_nnz)) {
          base_dom_map_[var_nnz] = arith::IntSet::FromMinExtent(Integer(1), axis->nnz);
        }
      }
      if (!axis->IsVariable()) {
        const VarNode* var_nnz_cols = axis->nnz_cols.value().as<VarNode>();
        if (var_nnz_cols) {
          if (!base_dom_map_.count(var_nnz_cols)) {
            base_dom_map_[var_nnz_cols] =
                arith::IntSet::FromMinExtent(Integer(1), axis->nnz_cols.value());
          }
        }
      }
    }
  }

  /*!
   * \brief Generated the loop nests for the outside the input body.
   * \param body The statement to be wrapped by loop nests.
   * \param block_iters The block iterators defined in the outermost block in `body`.
   * \param iter_binding The itervar bindings defined in the outermost block in `body`.
   * \param block_axes The axes corresponding to itervars defined in the outermost block in `body`.
   * \return The outermost generated loop.
   */
  Stmt GenerateLoops(Stmt body, const Array<IterVar>& block_iters,
                     const Array<PrimExpr>& iter_bindings, const Array<Axis>& block_axes) {
    int n_iter = static_cast<int>(block_iters.size());
    for (int i = n_iter - 1; i >= 0; --i) {
      const IterVar& iter_var = block_iters[i];
      if (!iter_bindings[i]->IsInstance<VarNode>()) {
        // skip if iter_binding is not a var (only happens in fused axis).
        continue;
      }
      const Var& loop_var = Downcast<Var>(iter_bindings[i]);
      const Axis& axis = block_axes[i];
      const Range& dom = iter_var->dom;
      PrimExpr extent = dom->extent;
      Optional<Buffer> maybe_indptr_buf;
      bool is_attached_axis = false;
      if (const AttachedAxisNode* attached_axis = axis.as<AttachedAxisNode>()) {
        maybe_indptr_buf = axis_indptr_map_.Get(attached_axis->base);
        is_attached_axis = true;
      } else {
        maybe_indptr_buf = axis_indptr_map_.Get(axis);
      }
      if (axis->IsVariable()) {
        ICHECK(maybe_indptr_buf.defined());
        Buffer indptr_buf = maybe_indptr_buf.value();
        Array<PrimExpr> indices;
        Array<Axis> ancestors =
            is_attached_axis ? CollectAncestors(axis, 1) : CollectAncestors(axis);
        for (const Axis& anc_axis : ancestors) {
          PrimExpr index = ctx_.GetIterVarFromAxis(anc_axis).value()->var;
          if (is_attached_axis) {
            index = VisitExpr(index);  // get coordinate for attached axis.
          }
          indices.push_back(index);
        }
        PrimExpr lb = BufferLoad(indptr_buf, indices);
        indices.Set(indices.size() - 1, indices.back() + 1);
        PrimExpr ub = BufferLoad(indptr_buf, indices);
        extent = ub - lb;
      } else {
        extent = axis->nnz_cols.value();
      }
      body = For(loop_var, Integer(0), extent, ForKind::kSerial, std::move(body));
    }
    return body;
  }

  /*! \brief Visitor of sparse iteration node.
   *  \return The emitted lowered block corresponding to the original sparse iteration.
   */
  Stmt VisitStmt_(const SparseIterationNode* sp_iteration) final {
    /*! \brief A class temporarily storing the block signatures and the outer loop variables of the
     * blocks to be generated */
    struct BlockInfo {
      /*! \brief The iterators of the block */
      Array<IterVar> block_iters;
      /*! \brief The axes appeared in the block */
      Array<Axis> block_axes;
      /*! \brief The loop vars in the block */
      Array<PrimExpr> iter_bindings;
      /*! \brief The init statement of the block */
      Optional<Stmt> init;

      /*!
       * \brief Push a new block iterator/iterator binding/axis to this block.
       * \param block_iter The block iterator to be pushed.
       * \param iter_binding The iterator binding to be pushed.
       * \param block_axis The axis to be pushed.
       */
      void Push(const IterVar& block_iter, const PrimExpr& iter_binding, const Axis& block_axis) {
        block_iters.push_back(block_iter);
        iter_bindings.push_back(iter_binding);
        block_axes.push_back(block_axis);
      }

      /*!
       * \brief Check whether a new block is needed. We need to create a new block when:
       * - the input axis is variable (dense-variable or sparse-variable), and
       * - the parent axis of the input axis has corresponding loop variable in the current block.
       * \param axis The axis to be checked.
       * \return Whether a new block is needed according to the conditions above.
       */
      bool NeedCreateNewBlock(LowerSparseIterContext* ctx, Axis axis) {
        if (!axis->IsVariable()) {
          // is fixed axis.
          return false;
        }

        Axis parent_axis = GetParentAxis(axis);
        Optional<SpIterVar> parent_iter_var = ctx->GetIterVarFromAxis(parent_axis);
        CHECK(parent_iter_var.defined())
            << "ValueError: The parent axis of " << axis << " does not appear.";

        for (const Axis& blk_axis : block_axes) {
          if (GetAxisBeforeFuse(blk_axis).same_as(parent_axis)) {
            return true;
          }
        }

        return false;
      }
    };

    int n_iters = static_cast<int>(sp_iteration->sp_iter_vars.size());
    Array<Var> loop_vars;

    // Enter the context
    ctx_.EnterScope(base_dom_map_);

    // Create the new loop variables, and update axis_itervar and var_itervar map in the
    // context.
    for (const SpIterVar& sp_iter_var : sp_iteration->sp_iter_vars) {
      Var loop_var("_" + sp_iter_var->var->name_hint, sp_iter_var->var->dtype);
      loop_vars.push_back(loop_var);
      ctx_.AddAxisIterVar(GetAxisBeforeFuse(sp_iter_var->axis), sp_iter_var);
      ctx_.AddVarIterVar(sp_iter_var->var, sp_iter_var);
    }

    // Mutate the `init` field.
    Optional<Stmt> init = sp_iteration->init.defined() ? VisitStmt(sp_iteration->init.value())
                                                       : Optional<Stmt>(NullOpt);

    // Gather the information of the blocks to be generated.
    std::vector<BlockInfo> block_infos(1);
    /* Whether a reduction block iterator has appeared */
    bool has_reduction_var = false;

    for (int i = 0; i < n_iters; ++i) {
      SpIterVar sp_iter_var = sp_iteration->sp_iter_vars[i];
      if (block_infos.back().NeedCreateNewBlock(&ctx_, sp_iter_var->axis)) {
        // Create a new BlockInfo;
        block_infos.emplace_back();
      }
      // Create loop information
      bool remove_loop_var = false;
      if (const FusedAxisNode* fused_axis = sp_iter_var->axis.as<FusedAxisNode>()) {
        // if it's fused axis, and not the last fused axis, remove the loop var.
        if (!fused_axis->IsLastAxis()) {
          remove_loop_var = true;
        }
      }
      PrimExpr iter_binding = remove_loop_var ? Integer(0) : PrimExpr(loop_vars[i]);
      block_infos.back().Push(sp_iter_var->as_iter_var(), iter_binding, sp_iter_var->axis);
      if (!has_reduction_var && sp_iter_var->is_reduction) {
        block_infos.back().init = std::move(init);
        has_reduction_var = true;
      }
    }

    // Recursively mutate the block body.
    Stmt body = VisitStmt(sp_iteration->body);

    // Generate nested blocks and loops from innermost to outermost.
    for (int i = static_cast<int>(block_infos.size()) - 1; i >= 0; --i) {
      BlockInfo info = std::move(block_infos[i]);

      // Collect read/write regions.
      ctx_.CollectRegion(true);  // update is_collecting_regions flag to true;
      Optional<Stmt> init = NullOpt;
      if (info.init.defined()) {
        init = VisitStmt(info.init.value());
      }
      body = VisitStmt(body);

      // Update read/writes regions.
      Array<BufferRegion> writes_new = ctx_.CollectWriteRegions();
      std::unordered_set<const BufferNode*> excluded_buffers;
      bool is_reduction = false;
      for (const IterVar& iter_var : info.block_iters) {
        if (iter_var->iter_type == kCommReduce) {
          is_reduction = true;
        }
      }
      if (is_reduction) {
        for (const BufferRegion& write_access : writes_new) {
          excluded_buffers.insert(write_access->buffer.get());
        }
      }
      Array<BufferRegion> reads = ctx_.CollectReadRegions(), reads_new;
      for (const BufferRegion& read_access : reads) {
        if (!excluded_buffers.count(read_access->buffer.get())) {
          reads_new.push_back(read_access);
        }
      }
      ctx_.ClearReadWriteBufferRegions();
      ctx_.CollectRegion(false);  // update is_collecting_regions flag to false

      // Create new block.
      Block block(/*iter_vars=*/info.block_iters,
                  /*reads=*/reads_new,
                  /*writes=*/writes_new,
                  /*name_hint=*/sp_iteration->name + std::to_string(i),
                  /*body=*/body,
                  /*init=*/init,
                  /*alloc_buffers=*/{},
                  /*match_buffers=*/{},
                  /*annotations=*/{{"sparse", Bool(true)}});

      // Update var dom.
      for (const IterVar& iter_var : info.block_iters) {
        ctx_.AddVarDom(iter_var->var, arith::IntSet::FromRange(iter_var->dom));
      }

      // Create block realize node.
      BlockRealize block_realize(/*iter_values=*/info.iter_bindings,
                                 /*predicate=*/const_true(),
                                 /*block=*/std::move(block));

      // Create loops
      body = std::move(block_realize);
      std::vector<BlockRealize>& bsearch_blocks = ctx_.GetBinarySearchBlocks();
      if (!bsearch_blocks.empty()) {
        Array<Stmt> blocks =
            RewriteBinarySearchBlocks(bsearch_blocks, info.block_iters, info.iter_bindings);
        blocks.push_back(std::move(body));
        body = SeqStmt(blocks);
        std::vector<std::tuple<Buffer, Var, Range>>& bsearch_ranges = ctx_.GetBinarySearchRanges();
        for (auto& it : bsearch_ranges) {
          Buffer buf;
          Var var;
          Range range;
          std::tie(buf, var, range) = it;
          ctx_.AddVarDom(var, arith::IntSet::FromRange(range));
        }
        bsearch_blocks.clear();  // only insert the blocks once.
      }
      Stmt loop = GenerateLoops(body, info.block_iters, info.iter_bindings, info.block_axes);
      body = std::move(loop);
    }

    // Exit the context.
    ctx_.ExitScope();

    return body;
  }

  /*! \brief Visitor of block realize node, used to collect read/write regions. */
  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    if (op->block->name_hint == "root") {
      // root block, collect alloc buffers
      for (const Buffer& buf : op->block->alloc_buffers) {
        root_alloc_buffers.push_back(std::move(buf));
      }
      return VisitStmt(op->block->body);
    }
    BufVarRewriter rewrite(&ctx_);
    /*! \note detector will not visit child block recursively, so it will stop here */
    for (const BufferRegion& read_access : op->block->reads) {
      std::vector<arith::IntSet> relaxed_region;
      for (const auto& range : read_access->region) {
        relaxed_region.push_back(arith::EvalSet(arith::IntSet::FromRange(Range::FromMinExtent(
                                                    rewrite(range->min), rewrite(range->extent))),
                                                ctx_.GetDomMap()));
      }
      ctx_.UpdateRead(read_access->buffer, relaxed_region);
    }
    for (const BufferRegion& write_access : op->block->writes) {
      std::vector<arith::IntSet> relaxed_region;
      for (const auto& range : write_access->region) {
        relaxed_region.push_back(arith::EvalSet(arith::IntSet::FromRange(Range::FromMinExtent(
                                                    rewrite(range->min), rewrite(range->extent))),
                                                ctx_.GetDomMap()));
      }
      ctx_.UpdateWrite(write_access->buffer, relaxed_region);
    }
    return GetRef<BlockRealize>(op);
  }

  /*! \brief Visitor of variable node.
   *  \note return decompressed coodinates for itervars corresponding to sparse axes.
   */
  PrimExpr VisitExpr_(const VarNode* op) final {
    Var var = GetRef<Var>(op);
    if (!ctx_.IsCollectingRegions()) {
      // decompress variable
      Optional<SpIterVar> maybe_sp_iter_var = ctx_.GetSpIterVarFromVar(GetRef<Var>(op));
      if (maybe_sp_iter_var.defined()) {
        SpIterVar sp_iter_var = maybe_sp_iter_var.value();
        Axis axis = sp_iter_var->axis;
        if (const FusedAxisNode* fused_axis = axis.as<FusedAxisNode>()) {
          // handle the special case of fused axis.
          PrimExpr offset = ctx_.GetIterVarFromAxis(fused_axis->group.back()).value()->var;
          for (int i = fused_axis->group.size() - 1; i > fused_axis->index; i--) {
            Axis original_axis = GetAxisBeforeFuse(fused_axis->group[i]);
            Optional<Buffer> maybe_indptr_buf = axis_indptr_map_.Get(original_axis);
            ICHECK(maybe_indptr_buf.defined()) << "Not a variable axis.";
            Buffer indptr_buf = maybe_indptr_buf.value();
            Array<Axis> ancestors = CollectAncestors(GetParentAxis(original_axis));
            Array<PrimExpr> prefix_indices;
            for (const Axis& ancestor : ancestors) {
              prefix_indices.push_back(ctx_.GetIterVarFromAxis(ancestor).value()->var);
            }
            offset =
                BinarySearch(indptr_buf, prefix_indices, Integer(0),
                             GetParentAxis(original_axis)->nnz + Integer(1), offset, false, true);
          }
          Axis original_axis = GetAxisBeforeFuse(fused_axis->group[fused_axis->index]);
          if (!original_axis->IsSparse()) {
            // if dense, return offset
            return offset;
          } else {
            // if sparse, get indices accoring to offset.
            Optional<Buffer> maybe_indices_buf = axis_indices_map_.Get(original_axis);
            ICHECK(maybe_indices_buf.defined()) << "Not a sparse axis.";
            Buffer indices_buf = maybe_indices_buf.value();
            Array<Axis> ancestors = CollectAncestors(original_axis);
            Array<PrimExpr> indices;
            for (const Axis& ancestor : ancestors) {
              indices.push_back(ctx_.GetIterVarFromAxis(ancestor).value()->var);
            }
            indices.push_back(offset);
            return BufferLoad(indices_buf, indices);
          }
        } else {
          Optional<Buffer> maybe_indices_buf;
          bool is_attached_axis = false;
          if (const AttachedAxisNode* attached_axis = axis.as<AttachedAxisNode>()) {
            maybe_indices_buf = axis_indices_map_.Get(attached_axis->base);
            is_attached_axis = true;
          } else {
            maybe_indices_buf = axis_indices_map_.Get(axis);
          }
          if (maybe_indices_buf.defined()) {
            Buffer indices_buf = maybe_indices_buf.value();
            Array<Axis> ancestors =
                is_attached_axis ? CollectAncestors(axis, 1) : CollectAncestors(axis);
            Array<PrimExpr> indices;
            for (const Axis& anc_axis : ancestors) {
              indices.push_back(ctx_.GetIterVarFromAxis(anc_axis).value()->var);
            }
            indices.push_back(var);
            return BufferLoad(indices_buf, indices);
          } else {
            return var;
          }
        }
      } else {
        return var;
      }
    } else {
      // do nothing when collecting regions.
      return var;
    }
  }

  /*! \brief Get relaxed region of indices including variables. */
  std::vector<arith::IntSet> GetRelaxedRegion(Array<PrimExpr> indices) {
    std::vector<arith::IntSet> relaxed_region;
    for (const PrimExpr& index : indices) {
      relaxed_region.push_back(arith::EvalSet(arith::IntSet::Vector(index), ctx_.GetDomMap()));
    }
    return std::move(relaxed_region);
  }

  /*!
   * \brief Perform binary search inside TIR.
   * \return The buffer (size=1) containing the binary search result.
   */
  PrimExpr BinarySearch(Buffer buf, Array<PrimExpr> prefix_indices, PrimExpr lb, PrimExpr ub,
                        PrimExpr val, bool left, bool minus_one = false) {
    /* Algorithm:
     * - when left = true
     *   - pre-condition
     *     lb < ub, and the last dimension of buf is sorted.
     *   - loop-invariant
     *     low <= mid < high, buf[..., lb:low] < val, buf[..., high:ub] >= val
     *   - post-condition
     *     low = mid = high,  buf[..., lb:low] < val, buf[..., high:ub] >= val
     * - when left = false
     *   - pre-condition
     *     lb < ub, and the last dimension of buf is sorted.
     *   - loop-invariant
     *     low <= mid < high, buf[..., lb:low] <= val, buf[..., high:ub] > val
     *   - post-condition
     *     low = mid = high,  buf[..., lb:low] <= val, buf[..., high:ub] > val
     */
    ICHECK(buf->shape.size() == prefix_indices.size() + 1)
        << "The dimensionality of buffer shoule equal the length of prefix indices plus 1.";
    // Check bsearch_map_ to avoid duplicate searches.
    Array<ObjectRef> args;
    args.push_back(buf);
    args.push_back(prefix_indices);
    args.push_back(lb);
    args.push_back(ub);
    args.push_back(val);
    args.push_back(Bool(left));
    if (bsearch_map_.count(args)) {
      return bsearch_map_[args];
    }
    DataType dtype = buf->dtype;
    if (!bsearch_low.defined()) {
      bsearch_low = MakeScratchpad("low", dtype);
      root_alloc_buffers.push_back(bsearch_low.value());
    }
    if (!bsearch_high.defined()) {
      bsearch_high = MakeScratchpad("high", dtype);
      root_alloc_buffers.push_back(bsearch_high.value());
    }
    Buffer low = bsearch_low.value();
    Buffer high = bsearch_high.value();
    Buffer mid = MakeScratchpad("mid_" + std::to_string(bsearch_blk_counter), dtype);

    Stmt low_store = BufferStore(low, lb, {Integer(0)});
    Stmt high_store = BufferStore(high, ub, {Integer(0)});
    PrimExpr low_val = BufferLoad(low, {Integer(0)}), high_val = BufferLoad(high, {Integer(0)}),
             mid_val = BufferLoad(mid, {Integer(0)});
    PrimExpr while_cond = low_val < high_val;
    Stmt mid_store = BufferStore(mid, low_val + floordiv(high_val - low_val, 2), {Integer(0)});
    Array<PrimExpr> indices = prefix_indices;
    indices.push_back(mid_val);
    PrimExpr pivot = BufferLoad(buf, indices);
    PrimExpr pivot_cmp_cond = left ? (pivot < val) : (pivot > val);
    Stmt if_true = left ? BufferStore(low, mid_val + 1, {Integer(0)})
                        : BufferStore(high, mid_val, {Integer(0)});
    Stmt if_false = left ? BufferStore(high, mid_val, {Integer(0)})
                         : BufferStore(low, mid_val + 1, {Integer(0)});
    Stmt if_then_else = IfThenElse(pivot_cmp_cond, if_true, if_false);
    SeqStmt while_body({mid_store, if_then_else});
    Stmt while_ = While(while_cond, while_body);
    Array<Stmt> body_stmts({low_store, high_store, while_});
    if (minus_one) {
      body_stmts.push_back(
          BufferStore(mid, BufferLoad(mid, {Integer(0)}) - Integer(1), {Integer(0)}));
    }
    SeqStmt body(body_stmts);

    /* read/write region update. */
    Array<Range> regions;
    for (const PrimExpr& index : prefix_indices) {
      regions.push_back(Range::FromMinExtent(index, Integer(1)));
    }
    regions.push_back(Range::FromMinExtent(lb, ub - lb));
    Array<BufferRegion> reads = {BufferRegion(buf, regions)};
    Array<BufferRegion> writes = {BufferRegion(
        mid,
        {Range::FromMinExtent(Integer(0), Integer(1))})};  // hide accesses to low/high buffers.

    Block new_block(/*iter_vars=*/{},
                    /*reads=*/reads,
                    /*writes=*/writes,
                    /*name_hint=*/"binary_search_" + std::to_string(bsearch_blk_counter),
                    /*body=*/body,
                    /*init=*/{},
                    /*alloc_buffers=*/{},
                    /*match_buffers=*/{},
                    /*annotations=*/{{"sparse", Bool(true)}});

    bsearch_blk_counter++;
    root_alloc_buffers.push_back(mid);
    PrimExpr root_init_value = Integer(-1);
    root_stmts.push_back(BufferStore(mid, root_init_value, {Integer(0)}));
    BlockRealize block_realize({}, mid_val == root_init_value, std::move(new_block));
    ctx_.AddBinarySearch(block_realize, mid, Range::FromMinExtent(Integer(0), buf->shape.back()));
    return mid_val;
  }

  /*! \brief Rewrite binary search blocks, bind with itervars. */
  Array<Stmt> RewriteBinarySearchBlocks(const std::vector<BlockRealize>& block_realizes,
                                        const Array<IterVar>& iter_vars,
                                        const Array<PrimExpr>& iter_bindings) {
    Array<Stmt> new_block_realizes;
    for (const BlockRealize& block_realize : block_realizes) {
      Array<IterVar> new_iter_vars;
      std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> block_var_map;
      for (size_t i = 0; i < iter_vars.size(); ++i) {
        const IterVar& iter_var = iter_vars[i];
        Var new_var("ax" + std::to_string(i), iter_var->var->dtype);
        IterVar new_iter_var(iter_var->dom, new_var, IterVarType::kDataPar);
        block_var_map[iter_var->var] = new_var;
        new_iter_vars.push_back(std::move(new_iter_var));
      }
      ObjectPtr<BlockNode> block_node = CopyOnWrite(block_realize->block.get());
      block_node->iter_vars = std::move(new_iter_vars);
      block_node->body = Substitute(block_node->body, block_var_map);
      Array<BufferRegion> new_reads;
      for (const BufferRegion& read : block_node->reads) {
        new_reads.push_back(BufferRegion(read->buffer, Substitute(read->region, block_var_map)));
      }
      block_node->reads = std::move(new_reads);
      // NOTE(zihao): no need to modify writes.
      Block new_block = Block(block_node);

      ObjectPtr<BlockRealizeNode> block_realize_node = CopyOnWrite(block_realize.get());
      block_realize_node->block = std::move(new_block);
      block_realize_node->iter_values = iter_bindings;
      new_block_realizes.push_back(BlockRealize(block_realize_node));
    }
    return new_block_realizes;
  }

  /*! \brief Return indices viewed in a given buffer. */
  Array<PrimExpr> RewriteIndices(Buffer buf, Array<PrimExpr> old_indices) {
    Array<PrimExpr> new_indices;
    if (const SparseBufferNode* sp_buf = buf.as<SparseBufferNode>()) {
      // rewrite indices for a sparse buffer.
      std::unordered_map<Axis, PrimExpr, ObjectPtrHash, ObjectPtrEqual> new_indices_map;
      std::unordered_map<Axis, bool, ObjectPtrHash, ObjectPtrEqual> match_map;
      // compute match map
      for (size_t i = 0; i < old_indices.size(); ++i) {
        PrimExpr index = old_indices[i];
        Axis buf_axis = sp_buf->axes[i];
        bool match = false;
        if (const VarNode* var = index.as<VarNode>()) {
          Optional<SpIterVar> maybe_sp_iter_var = ctx_.GetSpIterVarFromVar(GetRef<Var>(var));
          if (maybe_sp_iter_var.defined()) {
            SpIterVar sp_iter_var = maybe_sp_iter_var.value();
            if (const FusedAxisNode* fused_axis = sp_iter_var->axis.as<FusedAxisNode>()) {
              if (fused_axis->IsLastAxis() &&
                  GetAxisBeforeFuse(sp_iter_var->axis).same_as(buf_axis)) {
                match = true;
                Array<Axis> ancestors = CollectAncestors(buf_axis);
                for (const Axis& ancestor : ancestors) {
                  // overwrite previous axes.
                  match_map[ancestor] = true;
                }
              }
            }
            if (sp_iter_var->axis.same_as(buf_axis)) {
              if (buf_axis->parent.defined()) {
                // if has parent axis, match[axis] = match[parent]
                Axis parent = Downcast<Axis>(buf_axis->parent);
                if (match_map.count(parent) && match_map[parent]) {
                  match = true;
                }
              } else {
                // if not parent axis, match is true
                match = true;
              }
            }
          }
        }
        match_map[buf_axis] = match;
      }
      for (size_t i = 0; i < old_indices.size(); ++i) {
        PrimExpr index = old_indices[i];
        PrimExpr new_index;
        Axis buf_axis = sp_buf->axes[i];
        bool match = match_map[buf_axis];
        if (match) {
          new_index = index;
        } else {
          PrimExpr coordinate = VisitExpr(index);
          Optional<Buffer> maybe_indices_buf = axis_indices_map_.Get(buf_axis);
          if (maybe_indices_buf.defined()) {
            // it's sparse axis.
            Buffer indices_buf = maybe_indices_buf.value();
            Array<Axis> ancestors = CollectAncestors(buf_axis);
            Array<PrimExpr> indices_path;
            for (const Axis& ancestor : ancestors) {
              CHECK(new_indices_map.count(ancestor))
                  << "The indices of axis " << ancestor << " not found.";
              indices_path.push_back(new_indices_map[ancestor]);
            }
            PrimExpr extent;
            if (buf_axis->IsVariable()) {
              Buffer indptr_buf = axis_indptr_map_.Get(buf_axis).value();
              PrimExpr lb = BufferLoad(indptr_buf, indices_path), last_index = indices_path.back();
              indices_path.Set(indices_path.size() - 1, last_index + 1);
              PrimExpr ub = BufferLoad(indptr_buf, indices_path);
              indices_path.Set(indices_path.size() - 1, last_index);  // set last index back.
              extent = ub - lb;
            } else {
              extent = buf_axis->nnz_cols.value();
            }
            new_index =
                BinarySearch(indices_buf, indices_path, Integer(0), extent, coordinate, true);
          } else {
            // it's dense axis.
            new_index = coordinate;
          }
        }
        new_indices.push_back(new_index);
        new_indices_map[buf_axis] = new_index;
      }
    } else {
      // rewrite indices for a non-sparse buffer.
      for (const PrimExpr& index : old_indices) {
        // insert coordinates as new_indices.
        new_indices.push_back(VisitExpr(index));
      }
    }
    return new_indices;
  }

  /*! \brief Visitor of buffer load node. */
  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    if (ctx_.IsCollectingRegions()) {
      // The second time we visit the node.
      ctx_.UpdateRead(op->buffer, std::move(GetRelaxedRegion(op->indices)));
      for (const PrimExpr& index : op->indices) {
        VisitExpr(index);  // touch indices to handle indirect memory access
      }
      return GetRef<BufferLoad>(op);
    } else {
      // The first time we visit the node.
      return BufferLoad(op->buffer, RewriteIndices(op->buffer, op->indices));
    }
  }

  /*! \brief Visitor of buffer store node. */
  Stmt VisitStmt_(const BufferStoreNode* op) final {
    if (ctx_.IsCollectingRegions()) {
      // The second time we visit the node.
      ctx_.UpdateWrite(op->buffer, std::move(GetRelaxedRegion(op->indices)));
      VisitExpr(op->value);  // touch values
      for (const PrimExpr& index : op->indices) {
        VisitExpr(index);  // touch indices to handle indirect memory access
      }
      return GetRef<BufferStore>(op);
    } else {
      // The first time we visit the node.
      PrimExpr value = VisitExpr(op->value);
      return BufferStore(op->buffer, value, RewriteIndices(op->buffer, op->indices));
    }
  }

  LowerSparseIterContext ctx_;          // auxilliary context information.
  Map<Axis, Buffer> axis_indptr_map_;   // axis to indptr buffer map.
  Map<Axis, Buffer> axis_indices_map_;  // axis to indices buffer map.
  std::unordered_map<const VarNode*, arith::IntSet> base_dom_map_;  // The base dom map.
  std::unordered_map<ObjectRef, PrimExpr, StructuralHash, StructuralEqual>
      bsearch_map_;         // The map storing existing binary search keys and values.
  int bsearch_blk_counter;  // Counter for generated binary search blocks.
};

PrimFunc LowerSparseIter(PrimFunc f) {
  // Only apply this pass to TIR that is not from TE schedules
  if (!IsFromLegacyTESchedule(f) && SparseTIRLevel(f) == 2) {
    PrimFuncNode* fptr = f.CopyOnWrite();
    // Step 1. Update the PrimFunc's buffer map.
    Map<Axis, Buffer> axis_indptr_map, axis_indices_map;
    std::tie(axis_indptr_map, axis_indices_map, fptr->buffer_map, fptr->sp_axes) =
        UpdateBufferMap(f);
    // Step 2. Lower iterations.
    IterTransformer lower_sparse(axis_indptr_map, axis_indices_map, fptr->sp_axes);
    Stmt body = lower_sparse(std::move(fptr->body));
    // Step 3. Wrap with root block.
    if (!lower_sparse.root_stmts.empty()) {
      Array<Stmt> stmts = std::move(lower_sparse.root_stmts);
      stmts.push_back(std::move(body));
      body = SeqStmt(stmts);
    }
    Block root_block({}, {}, {}, "root", body, NullOpt, lower_sparse.root_alloc_buffers);
    fptr->body = BlockRealize({}, const_true(), std::move(root_block));
    // Step 4. Lower sparse tir level.
    Map<String, ObjectRef> new_attr_dict = fptr->attrs->dict;
    new_attr_dict.Set("sparse_tir_level", Integer(1));
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
Pass LowerSparseIter() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return LowerSparseIter(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerSparseIter", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerSparseIter").set_body_typed(LowerSparseIter);

}  // namespace transform

}  // namespace tir
}  // namespace tvm