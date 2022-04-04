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
 * \file horizontal_fusion.cc
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

using support::StartsWith;

class ThreadTagExtentCollector : public StmtExprVisitor {
 public:
  explicit ThreadTagExtentCollector() {}
  Map<String, Integer> Collect(const PrimFuncNode* fptr) {
    thread_tag_extent_map_.clear();
    VisitStmt(fptr->body);
    return thread_tag_extent_map_;
  }

 private:
  Map<String, Integer> thread_tag_extent_map_;

  void VisitStmt_(const ForNode* op) final {
    StmtExprVisitor::VisitStmt_(op);
    if (op->kind == ForKind::kThreadBinding) {
      CHECK_EQ(Downcast<Integer>(op->min)->value, 0)
          << "The min value of the loop should be 0 to perform horizontal fusion.";
      Integer extent = Downcast<Integer>(op->extent);
      String thread_tag = op->thread_binding.value()->thread_tag;
      Optional<Integer> maybe_prev_extent = thread_tag_extent_map_.Get(thread_tag);
      if (maybe_prev_extent.defined()) {
        Integer prev_extent = maybe_prev_extent.value();
        if (thread_tag == "blockIdx.x") {
          // Fuse horizontally on blockIdx.x
          thread_tag_extent_map_.Set(thread_tag, Integer(prev_extent->value + extent->value));
        } else if (StartsWith(thread_tag, "blockIdx")) {
          LOG(FATAL) << "blockIdx.y/z is not allowed in horizontal fusion.";
        } else {
          // Padded to maximum possible extent for other threads.
          thread_tag_extent_map_.Set(thread_tag,
                                     Integer(std::max(prev_extent->value, extent->value)));
        }
      } else {
        thread_tag_extent_map_.Set(thread_tag, extent);
      }
    }
  }
};

class HorizontalFuser : public StmtExprMutator {
 public:
  explicit HorizontalFuser(Map<String, Integer> thread_tag_extent_map)
      : block_idx("block_idx"),
        blockIdx_x_accum_offset_(0),
        thread_tag_extent_map_(std::move(thread_tag_extent_map)) {
    predicate_stack_.push_back({});
  }

 private:
  PrimExpr VisitExpr_(const VarNode* op) final {
    if (var_substitution_map_.find(op) != var_substitution_map_.end()) {
      return var_substitution_map_[op];
    } else {
      return GetRef<Var>(op);
    }
  }

  Stmt VisitStmt_(const ForNode* op) final {
    // If this For is not thread binding attribute, return as usual.
    if (op->kind != ForKind::kThreadBinding) {
      return StmtExprMutator::VisitStmt_(op);
    }
    String thread_tag = op->thread_binding.value()->thread_tag;
    Integer original_extent = Downcast<Integer>(op->extent);
    if (thread_tag == "blockIdx.x") {
      predicate_stack_.back().push_back((block_idx >= blockIdx_x_accum_offset_) &&
                                        (block_idx < blockIdx_x_accum_offset_ + original_extent));
      var_substitution_map_[op->loop_var.get()] = block_idx - blockIdx_x_accum_offset_;
      blockIdx_x_accum_offset_ += original_extent->value;
      Stmt body = VisitStmt(op->body);
      predicate_stack_.back().pop_back();
      return body;
    } else {
      Integer new_extent = thread_tag_extent_map_.Get(thread_tag).value();
      Var loop_var = op->loop_var;

      auto n = CopyOnWrite(op);
      if (original_extent->value != new_extent->value) {
        n->extent = new_extent;
        predicate_stack_.back().push_back(loop_var < original_extent);
        n->body = VisitStmt(n->body);
        predicate_stack_.back().pop_back();
      } else {
        n->body = VisitStmt(n->body);
      }
      return For(n);
    }
  }

  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    PrimExpr new_predicate = op->predicate;
    for (const PrimExpr& predicate : predicate_stack_.back()) {
      new_predicate = new_predicate && predicate;
    }
    auto n = CopyOnWrite(op);
    n->predicate = new_predicate;
    predicate_stack_.push_back({});
    n->block = Downcast<Block>(VisitStmt(n->block));
    predicate_stack_.pop_back();
    return BlockRealize(n);
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    if (op->name_hint == "root") {
      // add an extra loop in root block.
      auto n = CopyOnWrite(op);
      Stmt body = VisitStmt(n->body);
      For new_loop(block_idx, Integer(0), thread_tag_extent_map_.Get("blockIdx.x").value(),
                   ForKind::kThreadBinding, body,
                   IterVar(NullValue<Range>(), Var(""), IterVarType::kThreadIndex, "blockIdx.x"));
      n->body = new_loop;
      return Block(n);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Var block_idx;
  int32_t blockIdx_x_accum_offset_;
  Map<String, Integer> thread_tag_extent_map_;
  std::unordered_map<const VarNode*, PrimExpr> var_substitution_map_;
  std::vector<std::vector<PrimExpr>> predicate_stack_;
};

PrimFunc HorizontalFusion(PrimFunc f) {
  // Only apply this pass to TIR that is not from TE schedules
  if (!IsFromLegacyTESchedule(f)) {
    PrimFuncNode* fptr = f.CopyOnWrite();
    // If the horizontal fuse flag was set to True, apply horizontal fuser.
    Optional<Bool> maybe_horizontal_fuse_flag = fptr->attrs.GetAttr<Bool>("horizontal_fuse");
    if (maybe_horizontal_fuse_flag.defined()) {
      if (maybe_horizontal_fuse_flag.value()->value == 1) {
        ThreadTagExtentCollector collector;
        Map<String, Integer> thread_tag_extent_map_ = collector.Collect(fptr);
        fptr->body = HorizontalFuser(std::move(thread_tag_extent_map_))(std::move(fptr->body));
      }
      Map<String, ObjectRef> new_attr_dict = fptr->attrs->dict;
      new_attr_dict.erase("horizontal_fuse");
      if (new_attr_dict.empty()) {
        fptr->attrs = NullValue<DictAttrs>();
      } else {
        fptr->attrs = DictAttrs(new_attr_dict);
      }
    }
    return f;
  } else {
    return f;
  }
}

namespace transform {

Pass HorizontalFusion() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return HorizontalFusion(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.HorizontalFusion", {});
}

TVM_REGISTER_GLOBAL("tir.transform.HorizontalFusion").set_body_typed(HorizontalFusion);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
