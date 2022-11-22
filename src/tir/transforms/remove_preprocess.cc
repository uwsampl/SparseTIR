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
 * \file remove_preprocess.cc
 */

#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "ir_utils.h"

namespace tvm {
namespace tir {

class PreprocessRemover : public StmtExprMutator {
 public:
  PreprocessRemover() {}
  Map<Var, Buffer> extra_buffer_map;
  /* NOTE(Zihao): extra_buffer_map do not preserve order, use another array to record order
   * information. */
  Array<Var> extra_buffer_vars;

 private:
  Stmt VisitStmt_(const BlockNode* op) final {
    if (op->name_hint == "root") {
      auto n = CopyOnWrite(op);
      for (const Buffer& buf : op->alloc_buffers) {
        root_alloc_buffers.insert(buf.get());
      }
      if (!op->body->IsInstance<SeqStmtNode>()) {
        // no preprocessing block.
        return GetRef<Block>(op);
      }
      SeqStmt body = Downcast<SeqStmt>(op->body);
      Array<Stmt> seq;
      for (const Stmt& stmt : body->seq) {
        inside_preprocess_blk_ = false;
        VisitStmt(stmt);
        if (!inside_preprocess_blk_) {
          seq.push_back(stmt);
        }
      }
      // NOTE(Zihao): do not use SeqStmt if sequence length is 1.
      if (seq.size() == 1) {
        n->body = seq[0];
      } else {
        n->body = SeqStmt(seq);
      }
      Array<Buffer> new_alloc_buffers;
      for (const Buffer& buf : op->alloc_buffers) {
        if (!buffers_to_materialize.count(buf.get())) {
          new_alloc_buffers.push_back(buf);
        } else {
          Var new_var(buf->name + "_ptr", DataType::Handle());
          extra_buffer_map.Set(new_var, buf);
          extra_buffer_vars.push_back(new_var);
        }
      }
      n->alloc_buffers = new_alloc_buffers;
      return Block(n);
    } else {
      if (op->annotations.count("preprocess")) {
        inside_preprocess_blk_ = true;
      }
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  Stmt VisitStmt_(const SparseIterationNode* op) final {
    if (op->annotations.count("preprocess")) {
      inside_preprocess_blk_ = true;
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    if (inside_preprocess_blk_) {
      buffers_to_materialize.insert(op->buffer.get());
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  bool inside_preprocess_blk_ = false;
  std::unordered_set<const BufferNode*> root_alloc_buffers;
  std::unordered_set<const BufferNode*> buffers_to_materialize;
};

PrimFunc RemovePreprocess(PrimFunc f) {
  if (!IsFromLegacyTESchedule(f)) {
    PrimFuncNode* fptr = f.CopyOnWrite();
    PreprocessRemover remover;
    fptr->body = remover(fptr->body);
    // insert extra parameters
    for (const Var& var : remover.extra_buffer_vars) {
      fptr->params.push_back(var);
      ICHECK(remover.extra_buffer_map.count(var))
          << "Internal error, extra_buffer_map do not have key " << var;
      fptr->buffer_map.Set(var, remover.extra_buffer_map.Get(var).value());
    }
    return f;
  } else {
    return f;
  }
}

namespace transform {

Pass RemovePreprocess() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return RemovePreprocess(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.RemovePreprocess", {});
}

TVM_REGISTER_GLOBAL("tir.transform.RemovePreprocess").set_body_typed(RemovePreprocess);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
