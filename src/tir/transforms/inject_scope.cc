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
 * \file inject_scope.cc
 */
#include <tvm/arith/analyzer.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "ir_utils.h"

namespace tvm {
namespace tir {

namespace {

class VarCollector : public StmtExprVisitor {
 public:
  std::unordered_set<const VarNode*> used;

 private:
  void VisitExpr_(const VarNode* op) final { used.insert(op); }
};

}  // namespace

class ScopeInjector : public StmtExprMutator {
 public:
  ScopeInjector() {}

 private:
  Stmt VisitStmt_(const ForNode* op) final {
    bool is_thread_binding = op->kind == ForKind::kThreadBinding;
    if (is_thread_binding) {
      outer_loops_.push_back(GetRef<For>(op));
    }
    Stmt ret = StmtExprMutator::VisitStmt_(op);
    if (is_thread_binding) {
      outer_loops_.pop_back();
    }
    return ret;
  }

  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    PrimExpr cond = Bool(true);
    arith::Analyzer ana;
    VarCollector collector;
    collector(GetRef<BlockRealize>(op));
    Stmt body = StmtExprMutator::VisitStmt_(op);
    if (op->block->annotations.count("atomic")) {
      for (const For& loop : outer_loops_) {
        if (!collector.used.count(loop->loop_var.get())) {
          cond = cond && (loop->loop_var == loop->min);
        }
      }
      if (ana.CanProveEqual(cond, Bool(true))) {
        return body;
      } else {
        return IfThenElse(ana.Simplify(cond), body);
      }
    } else {
      return body;
    }
  }

  Array<For> outer_loops_;
};

PrimFunc InjectScope(PrimFunc f) {
  if (!IsFromLegacyTESchedule(f)) {
    PrimFuncNode* fptr = f.CopyOnWrite();
    ScopeInjector injector;
    fptr->body = injector(fptr->body);
    return f;
  } else {
    return f;
  }
}

namespace transform {

Pass InjectScope() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return InjectScope(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.InjectScope", {});
}

TVM_REGISTER_GLOBAL("tir.transform.InjectScope").set_body_typed(InjectScope);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
