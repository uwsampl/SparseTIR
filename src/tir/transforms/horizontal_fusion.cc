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
  ThreadTagExtentCollector() {}
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
      ICHECK(op->thread_binding.defined())
          << "The thread binding of " << GetRef<For>(op) << " is undefined.";
      String thread_tag = op->thread_binding.value()->thread_tag;
      Optional<Integer> maybe_prev_extent = thread_tag_extent_map_.Get(thread_tag);
      if (maybe_prev_extent.defined()) {
        Integer prev_extent = maybe_prev_extent.value();
        if (thread_tag == "blockIdx.x") {
          // Fuse horizontally on blockIdx.x
          thread_tag_extent_map_.Set(thread_tag, Integer(prev_extent->value + extent->value));
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
      : blockIdx_x_accum_offset_(0), thread_tag_extent_map_(std::move(thread_tag_extent_map)) {
    InitThreadTagVarMap();
  }

 private:
  void InitThreadTagVarMap() {
    thread_tag_var_map_.Set("blockIdx.x", Var("block_idx_x"));
    thread_tag_var_map_.Set("blockIdx.y", Var("block_idx_y"));
    thread_tag_var_map_.Set("blockIdx.z", Var("block_idx_z"));
    thread_tag_var_map_.Set("threadIdx.x", Var("thread_idx_x"));
    thread_tag_var_map_.Set("threadIdx.y", Var("thread_idx_y"));
    thread_tag_var_map_.Set("threadIdx.z", Var("thread_idx_z"));
  }

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
    ICHECK(op->thread_binding.defined())
        << "The thread binding of " << GetRef<For>(op) << " is undefined.";
    String thread_tag = op->thread_binding.value()->thread_tag;
    Integer original_extent = Downcast<Integer>(op->extent);
    CHECK(thread_tag_var_map_.count(thread_tag)) << "Unrecognized thread tag: " << thread_tag;
    Var thread_var = thread_tag_var_map_.Get(thread_tag).value();
    if (thread_tag == "blockIdx.x") {
      Stmt body;
      var_substitution_map_[op->loop_var.get()] = thread_var - blockIdx_x_accum_offset_;
      body = IfThenElse((thread_var < blockIdx_x_accum_offset_ + original_extent),
                        VisitStmt(op->body));
      blockIdx_x_accum_offset_ += original_extent->value;
      return body;
    } else {
      Integer new_extent = thread_tag_extent_map_.Get(thread_tag).value();
      Stmt body;
      var_substitution_map_[op->loop_var.get()] = thread_var;
      if (original_extent->value != new_extent->value) {
        body = IfThenElse(thread_var < original_extent, VisitStmt(op->body));
      } else {
        body = VisitStmt(op->body);
      }
      return body;
    }
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    if (op->name_hint == "root") {
      // add an extra loop in root block.
      auto n = CopyOnWrite(op);
      Stmt body = VisitStmt(n->body);
      if (body->IsInstance<SeqStmtNode>()) {
        SeqStmt seq = Downcast<SeqStmt>(body);
        Stmt outer;
        for (int i = seq->seq.size() - 1; i >= 0; i--) {
          ICHECK(seq->seq[i]->IsInstance<IfThenElseNode>()) << "Not an IfThenElse statement.";
          IfThenElse stmt = Downcast<IfThenElse>(seq->seq[i]);
          Stmt inner = outer;
          outer = IfThenElse(stmt->condition, stmt->then_case, inner);
        }
        body = outer;
      }

      for (auto& kv : thread_tag_extent_map_) {
        String thread_tag = kv.first;
        PrimExpr extent = kv.second;
        Var thread_var = thread_tag_var_map_.Get(thread_tag).value();
        For new_loop(thread_var, Integer(0), extent, ForKind::kThreadBinding, body,
                     IterVar(NullValue<Range>(), Var(""), IterVarType::kThreadIndex, thread_tag));
        body = new_loop;
      }
      n->body = body;
      return Block(n);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  int32_t blockIdx_x_accum_offset_;
  Map<String, Integer> thread_tag_extent_map_;
  Map<String, Var> thread_tag_var_map_;
  std::unordered_map<const VarNode*, PrimExpr> var_substitution_map_;
};

PrimFunc HorizontalFusion(PrimFunc f) {
  // Only apply this pass to TIR that is not from TE schedules
  if (!IsFromLegacyTESchedule(f)) {
    PrimFuncNode* fptr = f.CopyOnWrite();
    // If the horizontal fuse flag was set to True, apply horizontal fuser.
    Optional<ObjectRef> maybe_horizontal_fuse_flag =
        fptr->attrs.GetAttr<ObjectRef>("horizontal_fuse");
    if (maybe_horizontal_fuse_flag.defined()) {
      ThreadTagExtentCollector collector;
      Map<String, Integer> thread_tag_extent_map_ = collector.Collect(fptr);
      fptr->body = HorizontalFuser(std::move(thread_tag_extent_map_))(std::move(fptr->body));
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
