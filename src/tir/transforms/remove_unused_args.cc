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
 * \file remove_unused_args.cc
 */

#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "ir_utils.h"

namespace tvm {
namespace tir {

class UnusedArgsRemover : public StmtExprVisitor {
 public:
  explicit UnusedArgsRemover() {}
  std::unordered_set<const VarNode*> used_vars;
  std::unordered_set<const BufferNode*> used_bufs;

 private:
  void VisitExpr_(const VarNode* op) final {
    used_vars.insert(op);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    used_bufs.insert(op->buffer.get());
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    used_bufs.insert(op->buffer.get());
    StmtExprVisitor::VisitStmt_(op);
  }
};

PrimFunc RemoveUnusedArgs(PrimFunc f) {
  if (!IsFromLegacyTESchedule(f)) {
    PrimFuncNode* fptr = f.CopyOnWrite();
    UnusedArgsRemover remover;
    remover(fptr->body);
    CHECK(fptr->sp_axes.empty()) << "Only applicable to non-sparse tir scripts.";
    Array<Var> new_params;
    Map<Var, Buffer> new_buf_map;
    for (const auto& kv : fptr->buffer_map) {
      const Var& var = kv.first;
      const Buffer& buf = kv.second;
      if (remover.used_bufs.count(buf.get())) {
        new_buf_map.Set(var, buf);
        remover.used_vars.insert(var.get());
      }
    }
    for (const Var& var : fptr->params) {
      if (remover.used_vars.count(var.get())) {
        new_params.push_back(var);
      }
    }
    fptr->params = new_params;
    fptr->buffer_map = new_buf_map;
    return f;
  } else {
    return f;
  }
}

namespace transform {

Pass RemoveUnusedArgs() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return RemoveUnusedArgs(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.RemovePreprocess", {});
}

TVM_REGISTER_GLOBAL("tir.transform.RemoveUnusedArgs").set_body_typed(RemoveUnusedArgs);

}  // namespace transform

}  // namespace tir
}  // namespace tvm