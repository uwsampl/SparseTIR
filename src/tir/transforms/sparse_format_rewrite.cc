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
 * \brief sparse_format_rewrite.cc
 */
#include <tvm/tir/analysis.h>
#include <tvm/tir/format_rewrite.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../../support/utils.h"
#include "../schedule/analysis.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

namespace {

Array<Var> UpdateParams(const Array<FormatRewriteRule>& format_rewrite_rules,
                        const Array<Var>& orig_params) {
  Array<Var> ret;
  for (const Var& param : orig_params) {
    ret.push_back(param);
  }
  for (const FormatRewriteRule& rule : format_rewrite_rules) {
    for (const Var& param : rule->new_format_desc->params) {
      ret.push_back(param);
    }
  }
  return ret;
}

Map<Var, Buffer> UpdateBufferMap(const Array<FormatRewriteRule>& format_rewrite_rules,
                                 const Map<Var, Buffer>& buffer_map) {
  Map<Var, Buffer> ret;
  for (const auto& kv : buffer_map) {
    ret.Set(kv.first, kv.second);
  }
  for (const FormatRewriteRule& rule : format_rewrite_rules) {
    for (const auto& kv : rule->new_format_desc->buffer_map) {
      ret.Set(kv.first, kv.second);
    }
  }
  return ret;
}

Array<Axis> UpdateSparseAxes(const Array<FormatRewriteRule>& format_rewrite_rules,
                             const Array<Axis>& sp_axes) {
  Array<Axis> ret;
  for (const Axis& axis : sp_axes) {
    ret.push_back(axis);
  }
  for (const FormatRewriteRule& rule : format_rewrite_rules) {
    for (const Axis& axis : rule->new_format_desc->sp_axes) {
      ret.push_back(axis);
    }
  }
  return ret;
}

}  // namespace

class SparseFormatRewriter : public StmtExprMutator {
 public:
  explicit SparseFormatRewriter(Array<FormatRewriteRule> format_rewrite_rules)
      : format_rewrite_rules_(std::move(format_rewrite_rules)) {
    // CHECK
  }

 private:
  Array<FormatRewriteRule> format_rewrite_rules_;
};

PrimFunc SparseFormatRewrite(Array<FormatRewriteRule> format_rewrite_rules, PrimFunc f) {
  // Only apply this pass to TIR that is not from TE schedules
  if (!IsFromLegacyTESchedule(f)) {
    SparseFormatRewriter rewriter(format_rewrite_rules);
    PrimFuncNode* fptr = f.CopyOnWrite();
    fptr->params = UpdateParams(format_rewrite_rules, f->params);
    fptr->body = rewriter(f->body);
    fptr->buffer_map = UpdateBufferMap(format_rewrite_rules, f->buffer_map);
    fptr->sp_axes = UpdateSparseAxes(format_rewrite_rules, f->sp_axes);
    return f;
  } else {
    return f;
  }
}

namespace transform {

Pass SparseFormatRewrite(Array<FormatRewriteRule> format_rewrite_rules) {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return SparseFormatRewrite(std::move(format_rewrite_rules), std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.SparseFormatRewrite", {});
}

TVM_REGISTER_GLOBAL("tir.transform.SparseFormatRewrite").set_body_typed(SparseFormatRewrite);

}  // namespace transform

}  // namespace tir
}  // namespace tvm