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

#include "../utils.h"

namespace tvm {
namespace tir {

/******** Error Classes ********/

class NotSingleWriteBlock : public ScheduleError {
 public:
  explicit NotSingleWriteBlock(IRModule mod, Buffer buffer, Array<StmtSRef> write_blocks)
      : mod_(std::move(mod)), buffer_(std::move(buffer)) {
    ICHECK_GT(write_blocks.size(), 1);
    write_blocks_.reserve(write_blocks.size());
    for (const StmtSRef& block_sref : write_blocks) {
      const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
      write_blocks_.push_back(GetRef<Block>(block));
    }
  }

  String FastErrorString() const final {
    return "ScheduleError: The buffer is allowed to be written by single block.";
  }

  String DetailRenderTemplate() const final {
    size_t k = write_blocks_.size();
    return "The buffer " + buffer_->name + " is expected to be written by single block, but got " +
           std::to_string(k) + " blocks who write it.";
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final {
    return {write_blocks_.begin(), write_blocks_.end()};
  }

 private:
  IRModule mod_;
  Buffer buffer_;
  Array<Block> write_blocks_;
};

/******** Helper Functions/Classes ********/

/*! \brief The auxiliary info used for the insertion point and content of the cache stage. */
struct CacheStageInfo {
  /*! \brief The buffer to be read. */
  Buffer read_buffer;
  /*! \brief The buffer to be written. */
  Buffer write_buffer;
  /*! \brief The buffer allocation to be inserted into the block signature. */
  Buffer alloc;
  /*! \brief The AST node whose body is where the cache stage should be inserted. */
  StmtSRef loc_sref;
  /*! \brief The index to insert the cache_read/cache_write stage. */
  size_t loc_pos;
  /*! \brief The cache_read/cache_write stage to be inserted. */
  Stmt cache_stage;
  /*! \brief The map used for ScheduleStateNode::Replace. */
  Map<Block, Block> block_reuse;
  /*! \brief annotation of cache stage block. */
  Map<String, ObjectRef> annotations;
};

struct ReverseCacheTouchedInfo {
  /* Whether read or write. */
  bool read;
  /* Touched loop variable related information. */
  Array<Var> loop_vars;
  Array<Range> loop_ranges;
  /* Touched block variable related information. */
  Array<IterVar> block_vars;
  Array<PrimExpr> iter_values;
};

/*! \brief Return the buffer region related with the buffer */
Optional<BufferRegion> GetBufferRegionFromBuffer(const Array<BufferRegion>& buffer_regions,
                                                 const Buffer& buffer) {
  Optional<BufferRegion> res = NullOpt;
  for (const auto& region : buffer_regions) {
    if (region->buffer.same_as(buffer)) {
      ICHECK(!res.defined());
      res = region;
    }
  }
  return res;
}

/*!
 * \brief Create a loop nest that represents reverse cache copy (reverse_read / reverse_write) from
 * read buffer to write buffer. \param cache_region The cached copy region. \param info The cache
 * stage information, which will be updated in the function. \param storage_scope The storage scope
 * of the cached buffer (only used in naming here) \returns A block indicating the body of the loop
 * nesting.
 */
Block MakeReverseCacheStage(const BufferRegion& cache_region, ReverseCacheTouchedInfo* touched_info,
                            CacheStageInfo* info, const String& storage_scope) {
  /* Check whether cache region is a single point. */
  bool single_point = true;
  for (const Range& range : cache_region->region) {
    const auto* ext_int = range->extent.as<IntImmNode>();
    if (!ext_int || ext_int->value != 1) {
      single_point = false;
    }
  }
  CHECK(single_point) << "ReverseCacheStage is only valid when cache region is a single point.";
  // loop variables
  std::vector<Var> loop_vars;
  // block variables
  Array<IterVar> block_vars;
  // bindings in block realize
  std::vector<PrimExpr> iter_values;
  // Create loop vars and block vars' binding_value
  Map<Var, PrimExpr> var_map;
  for (size_t i = 0; i < touched_info->loop_vars.size(); ++i) {
    Var original_var = touched_info->loop_vars[i];
    Var loop_var("ax" + std::to_string(i), original_var.dtype());
    var_map.Set(original_var, loop_var);
    loop_vars.push_back(loop_var);
  }
  for (size_t i = 0; i < touched_info->block_vars.size(); ++i) {
    IterVar original_block_var = touched_info->block_vars[i];
    PrimExpr original_iter_value = touched_info->iter_values[i];
    IterVar block_var = IterVar(
        /*dom=*/original_block_var->dom,
        /*var=*/Var("v" + std::to_string(i), original_block_var->var.dtype()),
        /*IterVarType=*/kDataPar);
    var_map.Set(original_block_var->var, block_var->var);
    block_vars.push_back(block_var);
    iter_values.push_back(Substitute(original_iter_value, var_map));
  }

  // block access region for read/write buffers
  Region read_access_region, write_access_region;
  Array<PrimExpr> read_access_indices, write_access_indices;
  // Compute read/write region and read/write access indices.
  for (const Range& range : cache_region->region) {
    if (touched_info->read) {
      read_access_indices.push_back(Substitute(range->min, var_map));
      read_access_region.push_back(Range::FromMinExtent(read_access_indices.back(), Integer(1)));
    } else {
      write_access_indices.push_back(Substitute(range->min, var_map));
      write_access_region.push_back(Range::FromMinExtent(write_access_indices.back(), Integer(1)));
    }
  }
  for (const IterVar& block_var : block_vars) {
    if (touched_info->read) {
      write_access_indices.push_back(block_var->var);
      write_access_region.push_back(Range::FromMinExtent(write_access_indices.back(), Integer(1)));
    } else {
      read_access_indices.push_back(block_var->var);
      read_access_region.push_back(Range::FromMinExtent(read_access_indices.back(), Integer(1)));
    }
  }

  // Create New Block
  Block block(
      /*iter_vars*/ std::move(block_vars),
      /*reads=*/{BufferRegion(info->read_buffer, read_access_region)},
      /*writes=*/{BufferRegion(info->write_buffer, write_access_region)},
      /*name_hint*/ cache_region->buffer->name + "_" + storage_scope,
      /*body=*/
      BufferStore(info->write_buffer, BufferLoad(info->read_buffer, read_access_indices),
                  write_access_indices),
      /*init=*/NullOpt,
      /*alloc_buffers=*/{},
      /*match_buffers=*/{},
      /*buf_doms=*/{},
      /*annotations=*/info->annotations);
  // Create Block Realize node
  Stmt body = BlockRealize(/*values=*/iter_values,
                           /*predicate=*/const_true(),
                           /*block=*/block);
  // Create surrounding loops
  for (size_t i = loop_vars.size(); i >= 1; --i) {
    body = For(/*loop_var=*/loop_vars[i - 1],
               /*min=*/touched_info->loop_ranges[i - 1]->min,
               /*extent=*/touched_info->loop_ranges[i - 1]->extent,
               /*kind=*/ForKind::kSerial,
               /*body=*/body);
  }
  info->cache_stage = std::move(body);
  return block;
}

/*!
 * \brief Create a loop nest that represents cache copy (cache_read / cache_write) from read buffer
 *        to write buffer.
 * \note This function will store the stmt with loop nesting to the CacheStageInfo, but only return
 *        the inside block.
 * \param cache_region The cached copy region.
 * \param info The cache stage information, which will be updated in the function.
 * \param storage_scope The storage scope of the cached buffer (only used in naming here)
 * \returns A block indicating the body of the loop nesting.
 */
Block MakeCacheStage(const BufferRegion& cache_region, CacheStageInfo* info,
                     const String& storage_scope) {
  // loop variables
  std::vector<Var> loop_vars;
  // bindings in block realize
  std::vector<PrimExpr> iter_values;
  // Create loop vars and block vars' binding_value
  for (const Range& axis_range : cache_region->region) {
    Var loop_var("ax" + std::to_string(loop_vars.size()), axis_range->extent.dtype());
    loop_vars.push_back(loop_var);
    iter_values.push_back(axis_range->min + loop_var);
  }
  // block variables
  Array<IterVar> block_vars;
  // block access region for read/write buffers
  Region access_region;
  // indices used in block body
  Array<PrimExpr> access_indices;
  // Create block vars, block's accessed region and accessing indices
  for (const PrimExpr& dim : cache_region->buffer->shape) {
    Var var("v" + std::to_string(access_indices.size()), dim.dtype());
    block_vars.push_back(IterVar(/*dom=*/Range::FromMinExtent(0, dim),
                                 /*var=*/var,
                                 /*IterVarType=*/kDataPar));
    access_indices.push_back(var);
    access_region.push_back(Range::FromMinExtent(var, make_const(var.dtype(), 1)));
  }

  // Create the body block:
  //   reads = [read_buffer[access_region]]
  //   writes = [write_buffer[access_region]]
  //     write_buffer[access_indices] = read_buffer[access_indices]
  Block block(
      /*iter_vars=*/std::move(block_vars),
      /*reads=*/{BufferRegion(info->read_buffer, access_region)},
      /*writes=*/{BufferRegion(info->write_buffer, access_region)},
      /*name_hint=*/cache_region->buffer->name + "_" + storage_scope,
      /*body=*/
      BufferStore(info->write_buffer, BufferLoad(info->read_buffer, access_indices),
                  access_indices),
      /*init=*/NullOpt,
      /*alloc_buffers=*/{},
      /*match_buffers=*/{},
      /*buf_doms=*/{},
      /*annotations=*/info->annotations);
  // Create the block realize node
  Stmt body = BlockRealize(/*values=*/iter_values,
                           /*predicate=*/const_true(),
                           /*block=*/block);
  // Create surrounding loops
  for (size_t i = loop_vars.size(); i >= 1; --i) {
    body = For(/*loop_var=*/loop_vars[i - 1],
               /*min=*/0,
               /*extent=*/cache_region->region[i - 1]->extent,
               /*kind=*/ForKind::kSerial,
               /*body=*/body);
  }
  info->cache_stage = std::move(body);
  return block;
}

/*!
 * \brief Recalculate the `affine_binding` flag of a specifc block
 * \param block_sref The sref to the specific block
 */
bool CalculateAffineFlag(const ScheduleState& self, const StmtSRef& block_sref) {
  if (block_sref->parent == nullptr) {
    return true;
  }
  arith::Analyzer analyzer;
  analyzer.Bind(self->buf_dom_map);
  StmtSRef parent_sref = GetRef<StmtSRef>(block_sref->parent);
  return IsAffineBinding(/*realize=*/GetBlockRealize(self, block_sref),
                         /*loop_var_ranges=*/LoopDomainOfSRefTreePath(parent_sref),
                         /*analyzer=*/&analyzer);
}

/*!
 * \brief Insert the cache_read/cache_write stage into the specific position
 * \param stmt A sequence of statements or a single statement that the new stage is inserted in
 * \param pos The position where the cache stage is inserted
 * \param stage The stage to be inserted
 * \return A SeqStmt, the result after insertion
 */
SeqStmt InsertCacheStage(const Stmt& stmt, int pos, const Stmt& stage) {
  if (const auto* seq_stmt = stmt.as<SeqStmtNode>()) {
    ObjectPtr<SeqStmtNode> result = make_object<SeqStmtNode>(*seq_stmt);
    result->seq.insert(result->seq.begin() + pos, stage);
    return SeqStmt(result);
  }
  if (pos == 0) {
    return SeqStmt({stage, stmt});
  }
  ICHECK_EQ(pos, 1);
  return SeqStmt({stmt, stage});
}

/*!
 * \brief Get the only writer block of the input buffer in a given scope block.
 * \param self The state of the schedule
 * \param scope_sref The scope block where the write is considered
 * \param buffer The queried buffer
 * \return The sref of the only writer of the input buffer in the given scope,
 *         or `NullOpt` if no block writes it in the scope.
 * \throw NotSingleWriteBlock if there are more than one intrested block.
 */
Optional<StmtSRef> GetOnlyWriteBlock(ScheduleState self, const StmtSRef& scope_sref,
                                     const Buffer& buffer) {
  BlockScope scope = self->GetBlockScope(scope_sref);
  auto it = scope->buffer_writers.find(buffer);
  if (it == scope->buffer_writers.end()) {
    return NullOpt;
  } else {
    const Array<StmtSRef>& block_srefs = it->second;
    ICHECK(!block_srefs.empty());
    if (block_srefs.size() > 1) {
      throw NotSingleWriteBlock(self->mod, buffer, block_srefs);
    }
    return block_srefs[0];
  }
}

/*!
 * \brief Get the buffer region under the sref tree path [dom_low_inclusive, dom_high_exclusive)
 * \param self The state of the schedule.
 * \param buffer_region The buffer region to be analyzed.
 * \param block_sref The sref of the block related to the region.
 * \param dom_low_inclusive The lowest node in the sref tree path.
 * \param dom_high_exclusive The highest node in the sref tree path.
 * \return The relaxed buffer region.
 */
BufferRegion RelaxBufferRegion(ScheduleState self, const BufferRegion& buffer_region,
                               const StmtSRef& block_sref, const StmtSRef& dom_low_inclusive,
                               const StmtSRef& dom_high_exclusive) {
  BlockRealize realize = GetBlockRealize(self, block_sref);
  Map<Var, PrimExpr> binding = GetBindings(realize);
  const Buffer& buffer = buffer_region->buffer;
  arith::Analyzer analyzer;
  analyzer.Bind(self->buf_dom_map);
  BufferRegion subst_region = BufferRegion(buffer, Substitute(buffer_region->region, binding));
  Array<arith::IntSet> int_sets = AnalyzeRegionUpperBound(
      /*region=*/subst_region,
      /*predicate=*/realize->predicate,
      /*dom_low_inclusive=*/dom_low_inclusive,
      /*dom_high_exclusive=*/dom_high_exclusive,
      /*analyzer=*/&analyzer);
  ICHECK_EQ(buffer_region->region.size(), int_sets.size());

  Region region;
  region.reserve(int_sets.size());
  for (size_t i = 0; i < int_sets.size(); ++i) {
    region.push_back(int_sets[i].CoverRange(Range::FromMinExtent(0, buffer->shape[i])));
  }
  return BufferRegion(buffer, region);
}

/*! \brief Detect the insertion position of the new cache stage */
class CacheLocDetector : public StmtVisitor {
 public:
  /*!
   * \brief Detect the insertion position of the cache stage, and write the position into the
   * CacheStageInfo \param self The state of the schedule \param block_sref The sref of the unique
   * writer block of the buffer being applied cache_read or cache_write \param scope_sref The sref
   * of the scope block of the cached block \param info The cache stage info.
   */
  static void Detect(const ScheduleState& self, const StmtSRef& block_sref,
                     const StmtSRef& scope_sref, CacheStageInfo* info) {
    std::vector<StmtSRef> related_blocks;
    for (const Dependency& def : self->GetBlockScope(scope_sref)->GetDepsBySrc(block_sref)) {
      if (def->kind == DepKind::kRAW) {
        related_blocks.push_back(def->dst);
      }
    }
    if (!related_blocks.empty()) {
      CacheLocDetector detector(self, block_sref, scope_sref, related_blocks);
      detector(GetRef<Stmt>(scope_sref->stmt));
      info->loc_sref = detector.loc_sref_;
      info->loc_pos = detector.loc_pos_;
    } else {
      info->loc_sref = scope_sref;
      const auto* body = scope_sref->StmtAs<BlockNode>()->body.as<SeqStmtNode>();
      info->loc_pos = body == nullptr ? 1 : body->size();
    }
  }

 private:
  /*!
   * \brief Constructor
   * \param self The state of the schedule
   * \param block_sref The sref of the unique writer block of the buffer being applied cache_read or
   * cache_write \param scope_sref The sref of the scope block of the cached block \param
   * related_blocks Producer blocks for cache_write, or consumer blocks for cache_read
   */
  CacheLocDetector(const ScheduleState self, const StmtSRef& block_sref, const StmtSRef& scope_sref,
                   const std::vector<StmtSRef>& related_blocks)
      : self_(self),
        block_sref_(block_sref),
        scope_sref_(scope_sref),
        related_blocks_(related_blocks) {}

  void VisitStmt_(const SeqStmtNode* seq_stmt) final {
    bool previous_visited_block = visited_block_;
    bool previous_visited_related = visited_related_;
    visited_block_ = visited_related_ = false;

    int pos = -1;
    for (size_t i = 0; i < seq_stmt->size(); ++i) {
      if (loc_pos_ != -1) {
        break;
      }
      VisitStmt(seq_stmt->seq[i]);
      // `pos` can be assigned only once when we visited `block_sref`
      if (visited_block_ && visited_related_ && pos == -1) {
        // The offset of insert position from the block
        pos = i;
      }
    }
    visited_block_ = visited_block_ || previous_visited_block;
    visited_related_ = visited_related_ || previous_visited_related;
    // Only we visited the writing block and any one of the related blocks
    // That means that we have found the lowest ancestor
    // of the block and any one of the related ones
    if (visited_block_ && visited_related_ && loc_pos_ == -1) {
      loc_pos_ = pos;
    }
  }

  void VisitStmt_(const BlockNode* block) final {
    // Only visit the current scope under buffer writer's parent block
    if (block == scope_sref_->stmt) {
      // The block vistied is the current parent scope
      StmtVisitor::VisitStmt_(block);
      // Handling cache_read for input buffer
      if (visited_block_ && visited_related_ && !loc_sref_.defined()) {
        loc_sref_ = self_->stmt2ref.at(block);
        if (loc_pos_ == -1) {
          loc_pos_ = 1;
        }
      }
      return;
    }
    // Update `visited_block`
    if (block_sref_->stmt == block) {
      visited_block_ = true;
      return;
    }
    // Update `visited_related`
    for (const StmtSRef& related_block : related_blocks_) {
      if (related_block->stmt == block) {
        visited_related_ = true;
        return;
      }
    }
  }

  void VisitStmt_(const ForNode* loop) final {
    StmtVisitor::VisitStmt_(loop);
    if (visited_block_ && visited_related_ && !loc_sref_.defined() && loc_pos_ != -1) {
      loc_sref_ = self_->stmt2ref.at(loop);
    }
  }

 private:
  /*! \brief The schedule class */
  const ScheduleState self_;
  /*! \brief The dominate block which write the buffer */
  const StmtSRef& block_sref_;
  /*! \brief The parent scope of the dominate block */
  const StmtSRef& scope_sref_;
  /*! \brief Producer blocks for cache_write and consumer blocks for cache_read */
  const std::vector<StmtSRef>& related_blocks_;
  /*! \brief The flag whether we have visited the dominate block */
  bool visited_block_{false};
  /*! \brief The flag whether we have visited at least one related blocks */
  bool visited_related_{false};
  /*! \brief The AST node whose body is where the cache stage should be inserted */
  StmtSRef loc_sref_{nullptr};
  /*! \brief The index to insert the cache_read/cache_write stage */
  int loc_pos_{-1};
};

/*! \brief Mutator for ReverseCacheRead. */
class ReverseCacheReadRewriter : public StmtExprMutator {
 public:
  /*!
   * \brief Rewrite the AST and add a cache_read stage with the information provided.
   * \param scope_sref The parent scope of this mutation.
   * \param info The cache stage information.
   * \param touched_info The reverse cache touched information.
   * \return The new AST rooting at the original parent scope.
   */
  static Stmt Rewrite(const StmtSRef& scope_sref, CacheStageInfo* info,
                      ReverseCacheTouchedInfo* touched_info) {
    ReverseCacheReadRewriter rewriter(scope_sref, info, touched_info);
    return rewriter(GetRef<Stmt>(scope_sref->stmt));
  }

 private:
  explicit ReverseCacheReadRewriter(const StmtSRef& scope_sref, CacheStageInfo* info,
                                    ReverseCacheTouchedInfo* touched_info)
      : scope_sref_(scope_sref), info_(info) {
    for (const IterVar& iter_var : touched_info->block_vars) {
      new_indices_.push_back(iter_var->var);
    }
  }

  Stmt VisitStmt_(const ForNode* loop) final {
    Stmt stmt = StmtMutator::VisitStmt_(loop);
    // Check the insertion point
    if (loop == info_->loc_sref->stmt) {
      // Insert cache stage into the loop if it is the right place
      ObjectPtr<ForNode> n = make_object<ForNode>(*stmt.as<ForNode>());
      n->body = InsertCacheStage(n->body, info_->loc_pos, info_->cache_stage);
      stmt = Stmt(n);
    }
    return stmt;
  }

  Stmt VisitStmt_(const BlockNode* block) final {
    Block old_stmt = GetRef<Block>(block);
    if (block != scope_sref_->stmt &&
        GetBufferRegionFromBuffer(block->writes, info_->read_buffer).defined()) {
      return std::move(old_stmt);
    }
    // Mutate the body
    Block stmt = Downcast<Block>(StmtMutator::VisitStmt_(block));
    // Check the insertion point
    if (block == info_->loc_sref->stmt) {
      // Insert cache stage into the block if it is the right place
      ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
      n->body = InsertCacheStage(n->body, info_->loc_pos, info_->cache_stage);
      stmt = Block(n);
    }
    // Check if it is the block corresponding to the parent scope
    if (block == scope_sref_->stmt) {
      // If so, put buffer allocation on the parent scope
      ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
      n->alloc_buffers.push_back(info_->alloc);
      stmt = Block(n);
    } else {
      // Otherwise, update read regions and match_buffers
      Array<BufferRegion> reads;
      for (const BufferRegion& buf_region : block->reads) {
        if (buf_region->buffer.same_as(info_->read_buffer)) {
          Array<Range> region;
          for (const PrimExpr index : new_indices_) {
            region.push_back(Range::FromMinExtent(index, Integer(1)));
          }
          reads.push_back(BufferRegion(info_->write_buffer, region));
        } else {
          reads.push_back(buf_region);
        }
      }

      CHECK_EQ(block->match_buffers.size(), 0) << "Not supported yet.";
      if (!reads.same_as(block->reads)) {
        ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
        n->reads = std::move(reads);
        stmt = Block(n);
      }
    }
    info_->block_reuse.Set(old_stmt, stmt);
    return std::move(stmt);
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    if (op == info_->read_buffer->data.get()) {
      return info_->write_buffer->data;
    }
    return GetRef<PrimExpr>(op);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* load) final {
    if (load->buffer.same_as(info_->read_buffer)) {
      ObjectPtr<BufferLoadNode> n = make_object<BufferLoadNode>(*load);
      n->buffer = info_->write_buffer;
      n->indices = new_indices_;
      return PrimExpr(n);
    }
    return ExprMutator::VisitExpr_(load);
  }

  const StmtSRef& scope_sref_;
  CacheStageInfo* info_;
  Array<PrimExpr> new_indices_;
};

/*! \brief Mutator for CacheRead. */
class CacheReadRewriter : public StmtExprMutator {
 public:
  /*!
   * \brief Rewrite the AST and add a cache_read stage with the information provided
   * \param scope_sref The parent scope of this mutation
   * \param info The cache stage information
   * \return The new AST rooting at the original parent scope
   */
  static Stmt Rewrite(const StmtSRef& scope_sref, CacheStageInfo* info) {
    CacheReadRewriter rewriter(scope_sref, info);
    return rewriter(GetRef<Stmt>(scope_sref->stmt));
  }

 private:
  explicit CacheReadRewriter(const StmtSRef& scope_sref, CacheStageInfo* info)
      : scope_sref_(scope_sref), info_(info) {}

  Stmt VisitStmt_(const ForNode* loop) final {
    Stmt stmt = StmtMutator::VisitStmt_(loop);
    // Check the insertion point
    if (loop == info_->loc_sref->stmt) {
      // Insert cache stage into the loop if it is the right place
      ObjectPtr<ForNode> n = make_object<ForNode>(*stmt.as<ForNode>());
      n->body = InsertCacheStage(n->body, info_->loc_pos, info_->cache_stage);
      stmt = Stmt(n);
    }
    return stmt;
  }

  Stmt VisitStmt_(const BlockNode* block) final {
    Block old_stmt = GetRef<Block>(block);
    // We don't mutate the block which generates info->read_buffer
    if (block != scope_sref_->stmt &&
        GetBufferRegionFromBuffer(block->writes, info_->read_buffer).defined()) {
      return std::move(old_stmt);
    }
    // Mutate the body
    Block stmt = Downcast<Block>(StmtMutator::VisitStmt_(block));
    // Check the insertion point
    if (block == info_->loc_sref->stmt) {
      // Insert cache stage into the block if it is the right place
      ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
      n->body = InsertCacheStage(n->body, info_->loc_pos, info_->cache_stage);
      stmt = Block(n);
    }
    // Check if it is the block corresponding to the parent scope
    if (block == scope_sref_->stmt) {
      // If so, put buffer allocation on the parent scope
      ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
      n->alloc_buffers.push_back(info_->alloc);
      stmt = Block(n);
    } else {
      // Otherwise, update read regions and match_buffers
      Array<BufferRegion> reads =
          ReplaceBuffer(block->reads, info_->read_buffer, info_->write_buffer);
      Array<MatchBufferRegion> match_buffers =
          ReplaceBuffer(block->match_buffers, info_->read_buffer, info_->write_buffer);
      if (!reads.same_as(block->reads) || !match_buffers.same_as(block->match_buffers)) {
        ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
        n->reads = std::move(reads);
        n->match_buffers = std::move(match_buffers);
        stmt = Block(n);
      }
    }
    info_->block_reuse.Set(old_stmt, stmt);
    return std::move(stmt);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* load) final {
    if (load->buffer.same_as(info_->read_buffer)) {
      ObjectPtr<BufferLoadNode> n = make_object<BufferLoadNode>(*load);
      n->buffer = info_->write_buffer;
      return PrimExpr(n);
    }
    return ExprMutator::VisitExpr_(load);
  }

  PrimExpr VisitExpr_(const LoadNode* op) final {
    LOG(FATAL) << "Unexpected use of deprecated LoadNode.  Please use BufferLoadNode instead.";
    return PrimExpr();
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    if (op == info_->read_buffer->data.get()) {
      return info_->write_buffer->data;
    }
    return GetRef<PrimExpr>(op);
  }

 private:
  /*! \brief The parent scope of the insertion */
  const StmtSRef& scope_sref_;
  /*! \brief The info for inserting cache stage */
  CacheStageInfo* info_;
};

/*! \brief Mutator for ReverseCacheWrite. */
class ReverseCacheWriteRewriter : public StmtExprMutator {
 public:
  /*!
   * \brief Rewrite the AST and add a cache_write stage with the information provided.
   * \param scope_sref The parent scope of this mutation.
   * \param writer_block_sref The only writer block in the scope.
   * \param info The cache stage information.
   * \param touched_info The reverse cache touched information.
   * \return The new AST rooting at the original parent scope.
   */
  static Stmt Rewrite(const StmtSRef& scope_sref, const StmtSRef& writer_block_sref,
                      CacheStageInfo* info, ReverseCacheTouchedInfo* touched_info) {
    ReverseCacheWriteRewriter rewriter(scope_sref, writer_block_sref, info, touched_info);
    return rewriter(GetRef<Stmt>(scope_sref->stmt));
  }

 private:
  explicit ReverseCacheWriteRewriter(const StmtSRef& scope_sref, const StmtSRef& writer_block_sref,
                                     CacheStageInfo* info, ReverseCacheTouchedInfo* touched_info)
      : scope_sref_(scope_sref), writer_block_sref_(writer_block_sref), info_(info) {
    for (const IterVar& iter_var : touched_info->block_vars) {
      new_indices_.push_back(iter_var->var);
    }
  }

  Stmt VisitStmt_(const ForNode* loop) final {
    Stmt stmt = StmtMutator::VisitStmt_(loop);
    // Check the insertion point
    if (loop == info_->loc_sref->stmt) {
      // Insert cache stage into the loop if it is the right place
      ObjectPtr<ForNode> n = make_object<ForNode>(*stmt.as<ForNode>());
      n->body = InsertCacheStage(n->body, info_->loc_pos, info_->cache_stage);
      stmt = Stmt(n);
    }
    return stmt;
  }

  Stmt VisitStmt_(const BlockNode* block) final {
    Block old_stmt = GetRef<Block>(block);
    // We only mutate the block which generates info->write_buffer
    if (block != writer_block_sref_->stmt && block != scope_sref_->stmt && !under_writer_block_) {
      return std::move(old_stmt);
    }

    // Mutate the body
    bool under_scope = under_writer_block_ || block == writer_block_sref_->stmt;
    std::swap(under_scope, under_writer_block_);
    Block stmt = Downcast<Block>(StmtMutator::VisitStmt_(block));
    std::swap(under_scope, under_writer_block_);

    // Find the insertion point
    if (block == info_->loc_sref->stmt) {
      ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
      n->body = InsertCacheStage(n->body, info_->loc_pos, info_->cache_stage);
      stmt = Block(n);
    }
    // Put buffer allocation on the parent scope
    if (block == scope_sref_->stmt) {
      ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
      n->alloc_buffers.push_back(info_->alloc);
      stmt = Block(n);
    } else {
      // Since cache_write changes the block, we need to update the buffer it writes
      Array<BufferRegion> reads, writes;
      // New reads info.
      for (const BufferRegion& buf_region : block->reads) {
        if (buf_region->buffer.same_as(info_->write_buffer)) {
          Array<Range> region;
          for (const PrimExpr index : new_indices_) {
            region.push_back(Range::FromMinExtent(index, Integer(1)));
          }
          reads.push_back(BufferRegion(info_->read_buffer, region));
        } else {
          reads.push_back(buf_region);
        }
      }
      // New writes info, same as above.
      for (const BufferRegion& buf_region : block->writes) {
        if (buf_region->buffer.same_as(info_->write_buffer)) {
          Array<Range> region;
          for (const PrimExpr index : new_indices_) {
            region.push_back(Range::FromMinExtent(index, Integer(1)));
          }
          writes.push_back(BufferRegion(info_->read_buffer, region));
        } else {
          writes.push_back(buf_region);
        }
      }

      CHECK_EQ(block->match_buffers.size(), 0) << "Not supported yet.";
      if (!writes.same_as(block->writes) || !reads.same_as(block->reads)) {
        ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
        n->writes = std::move(writes);
        n->reads = std::move(reads);
        stmt = Block(n);
      }
    }
    // Remove atomic flag
    ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
    n->annotations.erase("atomic");
    stmt = Block(n);
    info_->block_reuse.Set(old_stmt, stmt);
    return std::move(stmt);
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    if (op == info_->write_buffer->data.get()) {
      return info_->read_buffer->data;
    }
    return GetRef<PrimExpr>(op);
  }

  Stmt VisitStmt_(const BufferStoreNode* store) final {
    BufferStore stmt = Downcast<BufferStore>(StmtMutator::VisitStmt_(store));
    if (stmt->buffer.same_as(info_->write_buffer)) {
      auto n = CopyOnWrite(stmt.get());
      n->buffer = info_->read_buffer;
      n->indices = new_indices_;
      return Stmt(n);
    } else {
      return std::move(stmt);
    }
  }

  PrimExpr VisitExpr_(const BufferLoadNode* load) final {
    if (load->buffer.same_as(info_->write_buffer)) {
      ObjectPtr<BufferLoadNode> n = make_object<BufferLoadNode>(*load);
      n->buffer = info_->read_buffer;
      n->indices = new_indices_;
      return PrimExpr(n);
    }
    return ExprMutator::VisitExpr_(load);
  }

  /*! \brief The parent scope of the insertion. */
  const StmtSRef& scope_sref_;
  /*! \brief The parent scope of the insertion. */
  const StmtSRef& writer_block_sref_;
  /*! \brief The info for inserting cache stage. */
  CacheStageInfo* info_;
  /*! \brief The indices to use for new buffer. */
  Array<PrimExpr> new_indices_;
  /*! \brief Whether the current node is under the given block. */
  bool under_writer_block_{false};
};

/*! \brief Mutator for CacheWrite */
class CacheWriteRewriter : public StmtExprMutator {
 public:
  /*!
   * \brief Rewrite the AST and add a cache_write stage with the information provided.
   * \param scope_sref The parent scope of this mutation.
   * \param writer_block_sref The only writer block in the scope.
   * \param info The cache stage information.
   * \return The new AST rooting at the original parent scope.
   */
  static Stmt Rewrite(const StmtSRef& scope_sref, const StmtSRef& writer_block_sref,
                      CacheStageInfo* info) {
    CacheWriteRewriter rewriter(scope_sref, writer_block_sref, info);
    return rewriter(GetRef<Stmt>(scope_sref->stmt));
  }

 private:
  explicit CacheWriteRewriter(const StmtSRef& scope_sref, const StmtSRef& writer_block_sref,
                              CacheStageInfo* info)
      : scope_sref_(scope_sref), writer_block_sref_(writer_block_sref), info_(info) {}

  Stmt VisitStmt_(const ForNode* loop) final {
    Stmt stmt = StmtMutator::VisitStmt_(loop);
    // Check the insertion point
    if (loop == info_->loc_sref->stmt) {
      // Insert cache stage into the loop if it is the right place
      ObjectPtr<ForNode> n = make_object<ForNode>(*stmt.as<ForNode>());
      n->body = InsertCacheStage(n->body, info_->loc_pos, info_->cache_stage);
      stmt = Stmt(n);
    }
    return stmt;
  }

  Stmt VisitStmt_(const BlockNode* block) final {
    Block old_stmt = GetRef<Block>(block);
    // We only mutate the block which generates info->write_buffer
    if (block != writer_block_sref_->stmt && block != scope_sref_->stmt && !under_writer_block_) {
      return std::move(old_stmt);
    }

    // Mutate the body
    bool under_scope = under_writer_block_ || block == writer_block_sref_->stmt;
    std::swap(under_scope, under_writer_block_);
    Block stmt = Downcast<Block>(StmtMutator::VisitStmt_(block));
    std::swap(under_scope, under_writer_block_);

    // Find the insertion point
    if (block == info_->loc_sref->stmt) {
      ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
      n->body = InsertCacheStage(n->body, info_->loc_pos, info_->cache_stage);
      stmt = Block(n);
    }
    // Put buffer allocation on the parent scope
    if (block == scope_sref_->stmt) {
      ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
      n->alloc_buffers.push_back(info_->alloc);
      stmt = Block(n);
    } else {
      // Since cache_write changes the block, we need to update the buffer it writes
      auto writes = ReplaceBuffer(block->writes, info_->write_buffer, info_->read_buffer);
      auto reads = ReplaceBuffer(block->reads, info_->write_buffer, info_->read_buffer);
      auto match_buffers =
          ReplaceBuffer(block->match_buffers, info_->write_buffer, info_->read_buffer);
      if (!writes.same_as(block->writes) || !reads.same_as(block->reads) ||
          !match_buffers.same_as(block->match_buffers)) {
        ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
        n->writes = std::move(writes);
        n->reads = std::move(reads);
        n->match_buffers = std::move(match_buffers);
        stmt = Block(n);
      }
    }
    // Remove atomic flag
    ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
    n->annotations.erase("atomic");
    stmt = Block(n);
    info_->block_reuse.Set(old_stmt, stmt);
    return std::move(stmt);
  }

  Stmt VisitStmt_(const BufferStoreNode* store) final {
    BufferStore stmt = Downcast<BufferStore>(StmtMutator::VisitStmt_(store));
    if (stmt->buffer.same_as(info_->write_buffer)) {
      auto n = CopyOnWrite(stmt.get());
      n->buffer = info_->read_buffer;
      return Stmt(n);
    } else {
      return std::move(stmt);
    }
  }

  PrimExpr VisitExpr_(const BufferLoadNode* load) final {
    if (load->buffer.same_as(info_->write_buffer)) {
      ObjectPtr<BufferLoadNode> n = make_object<BufferLoadNode>(*load);
      n->buffer = info_->read_buffer;
      return PrimExpr(n);
    }
    return ExprMutator::VisitExpr_(load);
  }

  PrimExpr VisitExpr_(const LoadNode* op) final {
    LOG(FATAL) << "Unexpected use of deprecated LoadNode.  Please use BufferLoadNode instead.";
    return PrimExpr();
  }

  Stmt VisitStmt_(const StoreNode* op) final {
    LOG(FATAL) << "Unexpected use of deprecated StoreNode.  Please use BufferStoreNode instead.";
    return Stmt();
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    if (op == info_->write_buffer->data.get()) {
      return info_->read_buffer->data;
    }
    return GetRef<PrimExpr>(op);
  }

 private:
  /*! \brief The parent scope of the insertion. */
  const StmtSRef& scope_sref_;
  /*! \brief The parent scope of the insertion. */
  const StmtSRef& writer_block_sref_;
  /*! \brief The info for inserting cache stage. */
  CacheStageInfo* info_;
  /*! \brief Whether the current node is under the given block. */
  bool under_writer_block_{false};
};

/******** Implementation ********/

StmtSRef CacheRead(ScheduleState self, const StmtSRef& block_sref, int read_buffer_index,
                   const String& storage_scope) {
  /*!
   * Check:
   *   - The index is in the array of block reading region
   *   - There is at most one block who write the buffer in the scope
   *
   * Mutate:
   *   - Allocate new cache buffer under the current scope.
   *   - Find the lowest ancestor of the block and ANY ONE of the consumers blocks.
   *   - Copy the buffer with the consumed region.
   */

  // Step 0. Check the input storage scope.
  CheckStorageScope(self, storage_scope);

  // Step 1. Check index, getting the target buffer and the parent scope
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  Buffer read_buffer =
      GetNthAccessBuffer(self, GetRef<Block>(block), read_buffer_index, /*is_write=*/false);
  StmtSRef scope_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/true);
  const BlockNode* scope_block = TVM_SREF_TO_BLOCK(scope_block, scope_sref);

  // Step 2. Create CacheStageInfo
  CacheStageInfo info;
  info.read_buffer = read_buffer;
  // Create the corresponding buffer to be written, i.e. result of cache_read
  info.write_buffer = WithScope(read_buffer, storage_scope);
  // Create the corresponding buffer allocation
  info.alloc = info.write_buffer;
  info.annotations = block->annotations;

  // Step 3. Update cache stage info.
  BufferRegion cache_region{nullptr};
  if (Optional<StmtSRef> _write_block_sref = GetOnlyWriteBlock(self, scope_sref, read_buffer)) {
    // Case 1. The buffer is written inside the block.
    StmtSRef write_block_sref = _write_block_sref.value();
    const BlockNode* write_block = TVM_SREF_TO_BLOCK(write_block, write_block_sref);
    // Find the producing region
    BufferRegion region = GetBufferRegionFromBuffer(write_block->writes, read_buffer).value();
    StmtSRef parent_sref = GetRef<StmtSRef>(write_block_sref->parent);

    // Detect insert position
    CacheLocDetector::Detect(self, write_block_sref, scope_sref, &info);
    cache_region = RelaxBufferRegion(self, region, write_block_sref, parent_sref, info.loc_sref);
  } else {
    // Case 2. The buffer is the input block for the scope.
    info.loc_sref = scope_sref;
    info.loc_pos = 0;
    if (Optional<BufferRegion> region =
            GetBufferRegionFromBuffer(scope_block->reads, read_buffer)) {
      cache_region = region.value();
    } else {
      cache_region = BufferRegion::FullRegion(read_buffer);
    }
  }

  // Step 4. Making new cache stage block and rewrite readers.
  Block cache_read_stage = MakeCacheStage(/*cache_region=*/cache_region, /*info=*/&info,
                                          /*storage_scope=*/storage_scope);
  Stmt new_scope = CacheReadRewriter::Rewrite(/*scope_sref=*/scope_sref, /*info=*/&info);

  // Step 5. Replacing and updating flags.
  self->Replace(scope_sref, new_scope, info.block_reuse);
  StmtSRef result_block_sref = self->stmt2ref.at(cache_read_stage.get());
  BlockInfo& block_info = self->block_info[result_block_sref];
  block_info.affine_binding = CalculateAffineFlag(self, result_block_sref);
  block_info.region_cover = true;
  block_info.scope->stage_pipeline = true;
  return result_block_sref;
}

StmtSRef CacheWrite(ScheduleState self, const StmtSRef& block_sref, int write_buffer_index,
                    const String& storage_scope) {
  /*!
   * Check:
   *   - The index is in the array of block reading region
   *   - There is only one block who write the buffer in the scope
   *
   * Mutate:
   *   - Allocate new cache buffer under the current scope.
   *   - Find the lowest ancestor of the block and ANY ONE of the producer blocks.
   *   - Copy the buffer with the consumed region.
   */

  // Step 0. Check the input storage scope.
  CheckStorageScope(self, storage_scope);

  // Step 1. Checking index, getting the target buffer and the parent scope
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  Buffer write_buffer =
      GetNthAccessBuffer(self, GetRef<Block>(block), write_buffer_index, /*is_write=*/true);
  StmtSRef scope_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/true);

  // Step 2. Creating CacheStageInfo
  CacheStageInfo info;
  info.read_buffer = WithScope(write_buffer, storage_scope);
  // Create the corresponding buffer to be written, i.e. result of cache_write
  info.write_buffer = write_buffer;
  // Create the corresponding buffer allocation
  info.alloc = info.read_buffer;
  info.annotations = block->annotations;

  // Step 3. Check the only writer block.
  if (!IsComposable(self, block_sref)) {
    ICHECK_EQ(block_sref.get(), GetOnlyWriteBlock(self, scope_sref, write_buffer).get());
  }

  // Step 4. Find the producing region and insert position
  BufferRegion region = GetBufferRegionFromBuffer(block->writes, write_buffer).value();
  StmtSRef parent_sref = GetRef<StmtSRef>(block_sref->parent);
  // Detect insert position
  CacheLocDetector::Detect(self, block_sref, scope_sref, &info);
  BufferRegion cache_region =
      RelaxBufferRegion(self, region, block_sref, parent_sref, info.loc_sref);

  // Step 5. Making new cache stage block and rewrite readers.
  Block cache_write_stage = MakeCacheStage(/*cache_region=*/cache_region, /*info=*/&info,
                                           /*storage_scope=*/storage_scope);
  Stmt new_scope = CacheWriteRewriter::Rewrite(/*scope_sref=*/scope_sref,
                                               /*writer_block_sref=*/block_sref, /*info=*/&info);

  // Step 6. Replacing and updating flags.
  self->Replace(scope_sref, new_scope, info.block_reuse);
  StmtSRef result_block_sref = self->stmt2ref.at(cache_write_stage.get());
  BlockInfo& block_info = self->block_info[result_block_sref];
  block_info.affine_binding = CalculateAffineFlag(self, result_block_sref);
  block_info.region_cover = true;
  block_info.scope->stage_pipeline = true;
  return result_block_sref;
}

class VarCollector : public ExprVisitor {
 public:
  explicit VarCollector() {}
  std::unordered_set<const VarNode*> touched;

 private:
  void VisitExpr_(const VarNode* op) final { touched.insert(op); }
};

Array<StmtSRef> GetLoopsUnderScope(const StmtSRef& block_sref, const StmtSRef& top_sref) {
  std::vector<StmtSRef> result;
  for (StmtSRefNode* parent = block_sref->parent; parent && parent->stmt->IsInstance<ForNode>();
       parent = parent->parent) {
    if (parent == top_sref.get()) break;
    result.push_back(GetRef<StmtSRef>(parent));
  }
  return {result.rbegin(), result.rend()};
}

StmtSRef ReverseCacheRead(ScheduleState self, const StmtSRef& block_sref, int read_buffer_index,
                          const String& storage_scope, Array<Integer> dim_order) {
  /*!
   * Check:
   *   - The index is in the array of block reading region
   *   - There is at most one block who write the buffer in the scope
   *
   * Mutate:
   *   - Allocate new cache buffer under the current scope.
   *   - Find the lowest ancestor of the block and ANY ONE of the consumers blocks.
   *   - Copy the buffer with the consumed region.
   */

  // Step 0. Check the input storage scope.
  CheckStorageScope(self, storage_scope);

  // Step 1. Check index, getting the target buffer and the parent scope
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  Buffer read_buffer =
      GetNthAccessBuffer(self, GetRef<Block>(block), read_buffer_index, /*is_write=*/false);
  StmtSRef scope_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/true);
  const BlockNode* scope_block = TVM_SREF_TO_BLOCK(scope_block, scope_sref);

  // Step 2. Create CacheStageInfo
  CacheStageInfo info;
  info.read_buffer = read_buffer;
  info.annotations = block->annotations;

  // Step 3. Update cache stage info.
  Optional<BufferRegion> maybe_region = GetBufferRegionFromBuffer(block->reads, read_buffer);
  ICHECK(maybe_region.defined()) << read_buffer
                                 << " should appear in the block's read region: " << block->reads;
  BufferRegion cache_region = maybe_region.value();
  if (Optional<StmtSRef> _write_block_sref = GetOnlyWriteBlock(self, scope_sref, read_buffer)) {
    // Case 1. The buffer is written inside the block.
    StmtSRef write_block_sref = _write_block_sref.value();
    const BlockNode* write_block = TVM_SREF_TO_BLOCK(write_block, write_block_sref);
    // Find the producing region
    StmtSRef parent_sref = GetRef<StmtSRef>(write_block_sref->parent);
    // Detect insert position
    CacheLocDetector::Detect(self, write_block_sref, scope_sref, &info);
  } else {
    // Case 2. The buffer is the input block for the scope.
    info.loc_sref = scope_sref;
    info.loc_pos = 0;
  }

  // Step 4. Create CacheTouchedInfo
  ReverseCacheTouchedInfo touched_info;

  // Step 5. Update CacheTouchedInfo
  touched_info.read = true;
  VarCollector collector;
  Array<PrimExpr> new_shape;
  for (const Range& range : cache_region->region) {
    collector(range->min);
  }
  BlockRealize realize = GetBlockRealize(self, block_sref);
  std::unordered_set<int> dim_order_set;
  for (const Integer& idx : dim_order) {
    dim_order_set.insert(idx->value);
  }
  for (size_t idx = 0; idx < block->iter_vars.size(); ++idx) {
    const IterVar& block_var = block->iter_vars[idx];
    if (collector.touched.count(block_var->var.get())) {
      if (dim_order_set.empty()) {
        // no user defined dim order.
        dim_order.push_back(idx);
      } else {
        // user provide dim order, check whether it's valid.
        CHECK(dim_order_set.count(idx))
            << "Block iter_var " << block_var
            << " used in the block, but doesn't appear in user-specified dim order array.";
      }
    }
  }

  for (size_t i = 0; i < dim_order.size(); ++i) {
    int idx = dim_order[i]->value;
    const IterVar& block_var = block->iter_vars[idx];
    touched_info.block_vars.push_back(block_var);
    touched_info.iter_values.push_back(realize->iter_values[idx]);
    new_shape.push_back(block_var->dom->min + block_var->dom->extent);
    collector(touched_info.iter_values.back());
  }

  for (const StmtSRef& loop_sref : GetLoopsUnderScope(block_sref, info.loc_sref)) {
    const ForNode* loop = TVM_SREF_TO_FOR(loop, loop_sref);
    if (collector.touched.count(loop->loop_var.get())) {
      touched_info.loop_vars.push_back(loop->loop_var);
      touched_info.loop_ranges.push_back(Range::FromMinExtent(loop->min, loop->extent));
    }
  }

  // Create write buffer.
  ObjectPtr<BufferNode> new_buffer = make_object<BufferNode>(*read_buffer.get());
  ObjectPtr<VarNode> new_var = make_object<VarNode>(*read_buffer->data.get());
  const auto* ptr_type = TVM_TYPE_AS(ptr_type, read_buffer->data->type_annotation, PointerTypeNode);
  new_var->type_annotation = PointerType(ptr_type->element_type, storage_scope);
  new_buffer->data = Var(new_var->name_hint + "_" + storage_scope, new_var->type_annotation);
  new_buffer->name = read_buffer->name + "_" + storage_scope;
  new_buffer->shape = new_shape;

  info.write_buffer = Buffer(new_buffer);
  info.alloc = info.write_buffer;

  // Step 6. Making new cache stage block and rewrite readers.
  Block cache_read_stage = MakeReverseCacheStage(/*cache_region=*/cache_region,
                                                 /*touched_info=*/&touched_info, /*info=*/&info,
                                                 /*storage_scope=*/storage_scope);
  Stmt new_scope = ReverseCacheReadRewriter::Rewrite(/*scope_sref=*/scope_sref, /*info=*/&info,
                                                     /*touched_info=*/&touched_info);

  // Step 7. Replacing and updating flags.
  self->Replace(scope_sref, new_scope, info.block_reuse);
  StmtSRef result_block_sref = self->stmt2ref.at(cache_read_stage.get());
  BlockInfo& block_info = self->block_info[result_block_sref];
  block_info.affine_binding = CalculateAffineFlag(self, result_block_sref);
  block_info.region_cover = true;
  block_info.scope->stage_pipeline = true;
  return result_block_sref;
}

StmtSRef ReverseCacheWrite(ScheduleState self, const StmtSRef& block_sref, int write_buffer_index,
                           const String& storage_scope, Array<Integer> dim_order) {
  /*!
   * Check:
   *   - The index is in the array of block reading region
   *   - There is only one block who write the buffer in the scope
   *
   * Mutate:
   *   - Allocate new cache buffer under the current scope.
   *   - Find the lowest ancestor of the block and ANY ONE of the producer blocks.
   *   - Copy the buffer with the consumed region.
   */

  // Step 0. Check the input storage scope.
  CheckStorageScope(self, storage_scope);

  // Step 1. Checking index, getting the target buffer and the parent scope
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  Buffer write_buffer =
      GetNthAccessBuffer(self, GetRef<Block>(block), write_buffer_index, /*is_write=*/true);
  StmtSRef scope_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/true);

  // Step 2. Creating CacheStageInfo
  CacheStageInfo info;
  info.write_buffer = write_buffer;
  info.annotations = block->annotations;

  // Step 3. Check the only writer block.
  if (!IsComposable(self, block_sref)) {
    ICHECK_EQ(block_sref.get(), GetOnlyWriteBlock(self, scope_sref, write_buffer).get());
  }

  // Step 4. Find the producing region and insert position
  Optional<BufferRegion> maybe_region = GetBufferRegionFromBuffer(block->writes, write_buffer);
  ICHECK(maybe_region.defined()) << write_buffer << " should appear in the block's write region";
  StmtSRef parent_sref = GetRef<StmtSRef>(block_sref->parent);
  // Detect insert position
  CacheLocDetector::Detect(self, block_sref, scope_sref, &info);
  BufferRegion cache_region = maybe_region.value();

  // Step 5. Create CacheTouchedInfo
  ReverseCacheTouchedInfo touched_info;

  // Step 6. Update CacheTouchedInfo
  touched_info.read = false;
  VarCollector collector;
  Array<PrimExpr> new_shape;
  for (const Range& range : cache_region->region) {
    collector(range->min);
  }
  BlockRealize realize = GetBlockRealize(self, block_sref);
  std::unordered_set<int> dim_order_set;
  for (const Integer& idx : dim_order) {
    dim_order_set.insert(idx->value);
  }
  for (size_t idx = 0; idx < block->iter_vars.size(); ++idx) {
    const IterVar& block_var = block->iter_vars[idx];
    if (collector.touched.count(block_var->var.get())) {
      if (dim_order_set.empty()) {
        // no user defined dim order.
        dim_order.push_back(idx);
      } else {
        // user provide dim order, check whether it's valid.
        CHECK(dim_order_set.count(idx))
            << "Block iter_var " << block_var
            << " used in the block, but doesn't appear in user-specified dim order array.";
      }
    }
  }

  for (size_t i = 0; i < dim_order.size(); ++i) {
    int idx = dim_order[i]->value;
    const IterVar& block_var = block->iter_vars[idx];
    touched_info.block_vars.push_back(block_var);
    touched_info.iter_values.push_back(realize->iter_values[idx]);
    new_shape.push_back(block_var->dom->min + block_var->dom->extent);
    collector(touched_info.iter_values.back());
  }

  for (const StmtSRef& loop_sref : GetLoopsUnderScope(block_sref, info.loc_sref)) {
    const ForNode* loop = TVM_SREF_TO_FOR(loop, loop_sref);
    if (collector.touched.count(loop->loop_var.get())) {
      touched_info.loop_vars.push_back(loop->loop_var);
      touched_info.loop_ranges.push_back(Range::FromMinExtent(loop->min, loop->extent));
    }
  }

  // Create write buffer.
  ObjectPtr<BufferNode> new_buffer = make_object<BufferNode>(*write_buffer.get());
  ObjectPtr<VarNode> new_var = make_object<VarNode>(*write_buffer->data.get());
  const auto* ptr_type =
      TVM_TYPE_AS(ptr_type, write_buffer->data->type_annotation, PointerTypeNode);
  new_var->type_annotation = PointerType(ptr_type->element_type, storage_scope);
  new_buffer->data = Var(new_var->name_hint + "_" + storage_scope, new_var->type_annotation);
  new_buffer->name = write_buffer->name + "_" + storage_scope;
  new_buffer->shape = new_shape;

  info.read_buffer = Buffer(new_buffer);
  info.alloc = info.read_buffer;

  // Step 7. Making new cache stage block and rewrite readers.
  Block cache_write_stage = MakeReverseCacheStage(/*cache_region=*/cache_region,
                                                  /*touched_info=*/&touched_info, /*info=*/&info,
                                                  /*storage_scope=*/storage_scope);
  Stmt new_scope = ReverseCacheWriteRewriter::Rewrite(
      /*scope_sref=*/scope_sref,
      /*writer_block_sref=*/block_sref, /*info=*/&info, /*touched_info=*/&touched_info);

  // Step 8. Replacing and updating flags.
  self->Replace(scope_sref, new_scope, info.block_reuse);
  StmtSRef result_block_sref = self->stmt2ref.at(cache_write_stage.get());
  BlockInfo& block_info = self->block_info[result_block_sref];
  block_info.affine_binding = CalculateAffineFlag(self, result_block_sref);
  block_info.region_cover = true;
  block_info.scope->stage_pipeline = true;
  return result_block_sref;
}

/******** Instruction Registration ********/

struct CacheReadTraits : public UnpackedInstTraits<CacheReadTraits> {
  static constexpr const char* kName = "CacheRead";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static BlockRV UnpackedApplyToSchedule(Schedule sch, BlockRV block, Integer read_buffer_index,
                                         String storage_scope) {
    return sch->CacheRead(block, read_buffer_index->value, storage_scope);
  }

  static String UnpackedAsPython(Array<String> outputs, String block, Integer read_buffer_index,
                                 String storage_scope) {
    PythonAPICall py("cache_read");
    py.Input("block", block);
    py.Input("read_buffer_index", read_buffer_index->value);
    py.Input("storage_scope", storage_scope);
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct CacheWriteTraits : public UnpackedInstTraits<CacheWriteTraits> {
  static constexpr const char* kName = "CacheWrite";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static BlockRV UnpackedApplyToSchedule(Schedule sch, BlockRV block, Integer write_buffer_index,
                                         String storage_scope) {
    return sch->CacheWrite(block, write_buffer_index->value, storage_scope);
  }

  static String UnpackedAsPython(Array<String> outputs, String block, Integer write_buffer_index,
                                 String storage_scope) {
    PythonAPICall py("cache_write");
    py.Input("block", block);
    py.Input("write_buffer_index", write_buffer_index->value);
    py.Input("storage_scope", storage_scope);
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct ReverseCacheReadTraits : public UnpackedInstTraits<ReverseCacheReadTraits> {
  static constexpr const char* kName = "ReverseCacheRead";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 3;
  static constexpr size_t kNumDecisions = 0;

  static BlockRV UnpackedApplyToSchedule(Schedule sch, BlockRV block, Integer read_buffer_index,
                                         String storage_scope, Array<Integer> dim_order) {
    return sch->ReverseCacheRead(block, read_buffer_index->value, storage_scope, dim_order);
  }

  static String UnpackedAsPython(Array<String> outputs, String block, Integer read_buffer_index,
                                 String storage_scope, Array<Integer> dim_order) {
    PythonAPICall py("reverse_cache_read");
    py.Input("block", block);
    py.Input("read_buffer_index", read_buffer_index->value);
    py.Input("storage_scope", storage_scope);
    py.Input("dim_order", dim_order);
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct ReverseCacheWriteTraits : public UnpackedInstTraits<ReverseCacheWriteTraits> {
  static constexpr const char* kName = "ReverseCacheWrite";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 3;
  static constexpr size_t kNumDecisions = 0;

  static BlockRV UnpackedApplyToSchedule(Schedule sch, BlockRV block, Integer write_buffer_index,
                                         String storage_scope, Array<Integer> dim_order) {
    return sch->ReverseCacheWrite(block, write_buffer_index->value, storage_scope, dim_order);
  }

  static String UnpackedAsPython(Array<String> outputs, String block, Integer write_buffer_index,
                                 String storage_scope, Array<Integer> dim_order) {
    PythonAPICall py("reverse_cache_write");
    py.Input("block", block);
    py.Input("write_buffer_index", write_buffer_index->value);
    py.Input("storage_scope", storage_scope);
    py.Input("dim_order", dim_order);
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(CacheReadTraits);
TVM_REGISTER_INST_KIND_TRAITS(CacheWriteTraits);
TVM_REGISTER_INST_KIND_TRAITS(ReverseCacheReadTraits);
TVM_REGISTER_INST_KIND_TRAITS(ReverseCacheWriteTraits);

}  // namespace tir
}  // namespace tvm
