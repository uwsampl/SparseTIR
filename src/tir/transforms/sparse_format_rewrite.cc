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

PrimFunc AddSuffix(PrimFunc func, String suffix) {
  auto* fptr = func.CopyOnWrite();
  // update params
  Map<Var, PrimExpr> var_map;
  Array<Var> new_params;
  for (const Var& var : func->params) {
    new_params.push_back(Var(var->name_hint + suffix, var->type_annotation));
    var_map.Set(var, new_params.back());
  }
  fptr->params = new_params;
  // update axes
  Array<Axis> new_axes;
  Map<Axis, Axis> axis_map;
  for (const Axis& axis : func->sp_axes) {
    Optional<ObjectRef> new_parent;
    if (axis->parent.defined()) {
      Axis old_parent = Downcast<Axis>(axis->parent.value());
      new_parent = axis_map.Get(old_parent).value();
    }
    Optional<PrimExpr> new_nnz_cols;
    if (axis->nnz_cols.defined()) {
      new_nnz_cols = Substitute(axis->nnz_cols.value(), var_map);
    }
    Optional<Var> new_indptr;
    if (axis->indptr.defined()) {
      new_indptr = Downcast<Var>(Substitute(axis->indptr.value(), var_map));
    }
    Optional<Var> new_indices;
    if (axis->indices.defined()) {
      new_indices = Downcast<Var>(Substitute(axis->indices.value(), var_map));
    }
    new_axes.push_back(Axis(axis->name + suffix, new_parent, Substitute(axis->length, var_map),
                            Substitute(axis->nnz, var_map), new_nnz_cols, new_indptr, new_indices,
                            axis->idtype));
    axis_map.Set(axis, new_axes.back());
  }
  fptr->sp_axes = new_axes;
  // update buffer_map
  Map<Var, Buffer> new_buffer_map;
  for (const auto& kv : func->buffer_map) {
    SparseBuffer old_buf = Downcast<SparseBuffer>(kv.second);
    Array<Axis> new_buf_axes;
    for (const Axis& buf_axis : old_buf->axes) {
      new_buf_axes.push_back(axis_map.Get(buf_axis).value());
    }
    new_buffer_map.Set(
        Downcast<Var>(Substitute(kv.first, var_map)),
        SparseBuffer(Downcast<Var>(Substitute(old_buf->data, var_map)), new_buf_axes,
                     old_buf->dtype, old_buf->name + suffix, old_buf->extra_storage));
  }
  fptr->buffer_map = new_buffer_map;
  return func;
}

Array<Var> UpdateParams(const Array<PrimFunc>& format_descs, const Array<Var>& orig_params) {
  Array<Var> ret;
  for (const Var& param : orig_params) {
    ret.push_back(param);
  }
  for (const PrimFunc& format_desc : format_descs) {
    for (const Var& param : format_desc->params) {
      ret.push_back(param);
    }
  }
  return ret;
}

Map<Var, Buffer> UpdateBufferMap(const Array<PrimFunc>& format_descs,
                                 const Map<Var, Buffer>& buffer_map) {
  Map<Var, Buffer> ret;
  for (const auto& kv : buffer_map) {
    ret.Set(kv.first, kv.second);
  }
  for (const PrimFunc& format_desc : format_descs) {
    for (const auto& kv : format_desc->buffer_map) {
      ret.Set(kv.first, kv.second);
    }
  }
  return ret;
}

Array<Axis> UpdateSparseAxes(const Array<PrimFunc>& format_descs, const Array<Axis>& sp_axes) {
  Array<Axis> ret;
  for (const Axis& axis : sp_axes) {
    ret.push_back(axis);
  }
  for (const PrimFunc& format_desc : format_descs) {
    for (const Axis& axis : format_desc->sp_axes) {
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
    Array<PrimFunc> format_descs;
    for (const FormatRewriteRule& rule : format_rewrite_rules) {
      format_descs.push_back(AddSuffix(rule->new_format_desc, "_" + rule->name));
    }
    fptr->params = UpdateParams(format_descs, f->params);
    fptr->body = rewriter(f->body);
    fptr->buffer_map = UpdateBufferMap(format_descs, f->buffer_map);
    fptr->sp_axes = UpdateSparseAxes(format_descs, f->sp_axes);
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