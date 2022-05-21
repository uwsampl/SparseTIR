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

String GetVarNameFromAxis(const Axis& axis) {
  std::string ret = axis->name;
  std::transform(ret.begin(), ret.end(), ret.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return ret;
}

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
    Var new_data(old_buf->name + suffix, old_buf->data->type_annotation);
    new_buffer_map.Set(Downcast<Var>(Substitute(kv.first, var_map)),
                       SparseBuffer(new_data, new_buf_axes, old_buf->dtype, old_buf->name + suffix,
                                    old_buf->extra_storage));
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

class IndexRewriter {
 public:
  void Init(Array<Axis> axes_before_rewrite, Array<Axis> axes_after_rewrite, IndexMap idx_map,
            IndexMap inv_idx_map) {
    axes_before_rewrite_ = std::move(axes_before_rewrite);
    axes_after_rewrite_ = std::move(axes_after_rewrite);
    idx_map_ = std::move(idx_map);
    inv_idx_map_ = std::move(inv_idx_map);
  }

  void UpdateInvMap(Map<Axis, PrimExpr>& axis_val_map) {
    Array<PrimExpr> after_indices;
    for (const Axis& axis : axes_after_rewrite_) {
      if (axis_val_map.count(axis)) {
        after_indices.push_back(axis_val_map.Get(axis).value());
      } else {
        after_indices.push_back(Integer(0));
      }
    }
    Array<PrimExpr> before_indices = inv_idx_map_->MapIndices(after_indices);
    for (size_t i = 0; i < axes_before_rewrite_.size(); ++i) {
      axis_val_map.Set(axes_before_rewrite_[i], before_indices[i]);
    }
  }

  void UpdateMap(Map<Axis, PrimExpr>& axis_val_map) {
    Array<PrimExpr> before_indices;
    for (const Axis& axis : axes_before_rewrite_) {
      if (axis_val_map.count(axis)) {
        before_indices.push_back(axis_val_map.Get(axis).value());
      } else {
        before_indices.push_back(Integer(0));
      }
    }
    Array<PrimExpr> after_indices = idx_map_->MapIndices(before_indices);
    for (size_t i = 0; i < axes_after_rewrite_.size(); ++i) {
      axis_val_map.Set(axes_after_rewrite_[i], after_indices[i]);
    }
  }

 private:
  Array<Axis> axes_before_rewrite_;
  Array<Axis> axes_after_rewrite_;
  IndexMap idx_map_;
  IndexMap inv_idx_map_;
};

class SparseFormatRewriter : public StmtExprMutator {
 public:
  explicit SparseFormatRewriter(const FormatRewriteRule& rule, const PrimFunc& new_func,
                                Array<Axis> old_axes, Array<SparseBuffer> old_buffers) {
    rewrite_suffix = "_" + rule->name;
    Array<Axis> old_axes_to_rewrite;
    for (const Axis& axis : old_axes) {
      name_axis_map_.Set(axis->name, axis);
      if (name_axis_map_.count(axis->name)) {
        old_axes_to_rewrite.push_back(axis);
      }
    }
    for (const Axis& axis : new_func->sp_axes) {
      name_axis_map_.Set(axis->name, axis);
    }
    for (const String& name : rule->axes_before_rewrite) {
      axes_before_rewrite_.push_back(name_axis_map_.Get(name).value());
    }
    for (const String& name : rule->axes_after_rewrite) {
      axes_after_rewrite_.push_back(name_axis_map_.Get(name + "_" + rule->name).value());
    }
    for (const auto& kv : rule->axis_map) {
      const Axis& k = name_axis_map_.Get(kv.first).value();
      Array<Axis> v;
      for (const String& name : kv.second) {
        v.push_back(name_axis_map_.Get(name + "_" + rule->name).value());
      }
      axis_rewrite_map_.Set(k, v);
    }
    for (const SparseBuffer& buf : old_buffers) {
      name_buf_map_.Set(buf->name, buf);
    }
    auto it = new_func->buffer_map.begin();
    for (size_t i = 0; i < rule->buffers_to_rewrite.size(); ++i, ++it) {
      String name = rule->buffers_to_rewrite[i];
      buffer_rewrite_map_.Set(name_buf_map_.Get(name).value(),
                              Downcast<SparseBuffer>((*it).second));
    }
    rewriter_.Init(axes_before_rewrite_, axes_after_rewrite_, rule->idx_map, rule->inv_idx_map);
    GenerateFormatRewriteBlock();
  }

  String rewrite_suffix;
  Array<Stmt> format_rewrites_blks;
  Array<Stmt> compute_blks;

 private:
  void GenerateFormatRewriteBlock() {
    for (const auto& kv : buffer_rewrite_map_) {
      const SparseBuffer& before_rewrite = kv.first;
      const SparseBuffer& after_rewrite = kv.second;

      Array<SpIterVar> sp_iter_vars;
      Array<PrimExpr> after_indices;
      Map<Axis, PrimExpr> axis_val_map;
      for (const Axis& axis : after_rewrite->axes) {
        Var var(GetVarNameFromAxis(axis), axis->idtype);
        after_indices.push_back(var);
        sp_iter_vars.push_back(SpIterVar(var, false, axis));
        axis_val_map.Set(axis, var);
      }
      rewriter_.UpdateInvMap(axis_val_map);
      Array<PrimExpr> before_indices;
      for (const Axis& axis : before_rewrite->axes) {
        before_indices.push_back(axis_val_map.Get(axis).value());
      }
      format_rewrites_blks.push_back(SparseIteration(
          sp_iter_vars, "rewrite_" + after_rewrite->name,
          BufferStore(after_rewrite, BufferLoad(before_rewrite, before_indices), after_indices),
          NullOpt, {{"preprocess", Bool(true)}}));
    }
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    CHECK(op->name_hint == "root")
        << "Cannot perform sparse format rewrite on a TVMScript with block other than root.";
    Stmt body = op->body;
    if (const SeqStmtNode* seq = body.as<SeqStmtNode>()) {
      // several sparse iterations
      for (const Stmt& sp_iter : seq->seq) {
        var_map_.clear();
        compute_blks.push_back(VisitStmt(sp_iter));
      }
    } else if (body->IsInstance<SparseIterationNode>()) {
      // one sparse iteration
      var_map_.clear();
      compute_blks.push_back(VisitStmt(body));
    } else {
      LOG(FATAL) << "Invalid root block body to rewrite";
    }
    return GetRef<Block>(op);
  }

  Stmt VisitStmt_(const SparseIterationNode* op) final {
    Array<SpIterVar> new_sp_iter_vars;
    Map<Var, PrimExpr> var_map;
    Map<Axis, PrimExpr> axis_val_map;
    for (const SpIterVar& sp_iter_var : op->sp_iter_vars) {
      if (axis_rewrite_map_.count(sp_iter_var->axis)) {
        Array<Axis> new_axes = axis_rewrite_map_.Get(sp_iter_var->axis).value();
        for (const Axis& axis : new_axes) {
          Var new_var(GetVarNameFromAxis(axis), sp_iter_var->var->dtype);
          SpIterVar new_sp_iter_var(new_var, sp_iter_var->is_reduction, axis);
          new_sp_iter_vars.push_back(new_sp_iter_var);
          axis_val_map.Set(axis, new_var);
          ana.Bind(new_var, Range::FromMinExtent(Integer(0), axis->length));
        }
      } else {
        Var new_var(sp_iter_var->var->name_hint, sp_iter_var->var->dtype);
        var_map.Set(sp_iter_var->var, new_var);
        SpIterVar new_sp_iter_var(new_var, sp_iter_var->is_reduction, sp_iter_var->axis);
        new_sp_iter_vars.push_back(new_sp_iter_var);
        axis_val_map.Set(sp_iter_var->axis, new_var);
        ana.Bind(new_var, Range::FromMinExtent(Integer(0), sp_iter_var->axis->length));
      }
    }
    rewriter_.UpdateInvMap(axis_val_map);
    for (const SpIterVar& sp_iter_var : op->sp_iter_vars) {
      var_map_.Set(sp_iter_var->var, axis_val_map.Get(sp_iter_var->axis).value());
    }

    Stmt body = VisitStmt(Substitute(op->body, var_map));
    Optional<Stmt> init = NullOpt;
    if (op->init.defined()) {
      init = VisitStmt(Substitute(op->init.value(), var_map));
    }
    return SparseIteration(new_sp_iter_vars, op->name + rewrite_suffix, body, init,
                           op->annotations);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    SparseBuffer buf = Downcast<SparseBuffer>(op->buffer);
    if (buffer_rewrite_map_.count(buf)) {
      Buffer new_buf = buffer_rewrite_map_.Get(buf).value();
      const SparseBufferNode* sp_buf = op->buffer.as<SparseBufferNode>();
      const SparseBufferNode* new_sp_buf = new_buf.as<SparseBufferNode>();
      Map<Axis, PrimExpr> axis_val_map;
      for (size_t i = 0; i < op->indices.size(); i++) {
        axis_val_map.Set(sp_buf->axes[i], VisitExpr(op->indices[i]));
      }
      rewriter_.UpdateMap(axis_val_map);
      rewriter_.UpdateInvMap(axis_val_map);
      Array<PrimExpr> new_indices;
      for (const Axis& axis : new_sp_buf->axes) {
        new_indices.push_back(ana.Simplify(axis_val_map.Get(axis).value()));
      }
      return BufferLoad(new_buf, new_indices);
    } else {
      Array<PrimExpr> new_indices;
      for (const PrimExpr& idx : op->indices) {
        new_indices.push_back(VisitExpr(idx));
      }
      return BufferLoad(op->buffer, new_indices);
    }
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    SparseBuffer buf = Downcast<SparseBuffer>(op->buffer);
    if (buffer_rewrite_map_.count(buf)) {
      SparseBuffer new_buf = buffer_rewrite_map_.Get(buf).value();
      Map<Axis, PrimExpr> axis_val_map;
      for (size_t i = 0; i < op->indices.size(); i++) {
        axis_val_map.Set(buf->axes[i], VisitExpr(op->indices[i]));
      }
      rewriter_.UpdateMap(axis_val_map);
      rewriter_.UpdateInvMap(axis_val_map);
      Array<PrimExpr> new_indices;
      for (const Axis& axis : new_buf->axes) {
        new_indices.push_back(ana.Simplify(axis_val_map.Get(axis).value()));
      }
      return BufferStore(new_buf, VisitExpr(op->value), new_indices);
    } else {
      Array<PrimExpr> new_indices;
      for (const PrimExpr& idx : op->indices) {
        new_indices.push_back(VisitExpr(idx));
      }
      return BufferStore(op->buffer, VisitExpr(op->value), new_indices);
    }
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    Var var = GetRef<Var>(op);
    if (var_map_.count(var)) {
      return var_map_.Get(var).value();
    } else {
      return var;
    }
  }

  Map<String, Axis> name_axis_map_;
  Map<String, SparseBuffer> name_buf_map_;
  Map<Axis, Array<Axis>> axis_rewrite_map_;
  Map<Var, PrimExpr> var_map_;
  Map<SparseBuffer, SparseBuffer> buffer_rewrite_map_;
  Array<Axis> axes_before_rewrite_;
  Array<Axis> axes_after_rewrite_;
  IndexRewriter rewriter_;
  arith::Analyzer ana;
};

PrimFunc SparseFormatRewrite(Array<FormatRewriteRule> format_rewrite_rules, PrimFunc f,
                             bool include_format_rewrite_blks = true) {
  // Only apply this pass to TIR that is not from TE schedules
  if (!IsFromLegacyTESchedule(f)) {
    // SparseFormatRewriter rewriter(format_rewrite_rules);
    PrimFuncNode* fptr = f.CopyOnWrite();
    Array<PrimFunc> format_descs;
    Array<Axis> old_sp_axes = f->sp_axes;
    Array<SparseBuffer> old_buffers;
    for (const auto& kv : f->buffer_map) {
      old_buffers.push_back(Downcast<SparseBuffer>(kv.second));
    }
    for (const FormatRewriteRule& rule : format_rewrite_rules) {
      format_descs.push_back(AddSuffix(rule->new_format_desc, "_" + rule->name));
    }
    fptr->params = UpdateParams(format_descs, f->params);
    fptr->buffer_map = UpdateBufferMap(format_descs, f->buffer_map);
    fptr->sp_axes = UpdateSparseAxes(format_descs, f->sp_axes);
    Array<Stmt> format_rewrite_blks, compute_blks;
    // generate format rewrite blocks and compute blocks for each rule
    for (size_t i = 0; i < format_rewrite_rules.size(); ++i) {
      SparseFormatRewriter rewriter(format_rewrite_rules[i], format_descs[i], old_sp_axes,
                                    old_buffers);
      rewriter(f->body);
      for (const Stmt& sp_iter : rewriter.format_rewrites_blks) {
        format_rewrite_blks.push_back(sp_iter);
      }
      for (const Stmt& sp_iter : rewriter.compute_blks) {
        compute_blks.push_back(sp_iter);
      }
    }
    // merge format rewrite and compute blocks.
    Array<Stmt> all_blks;
    if (include_format_rewrite_blks) {
      for (const Stmt& sp_iter : format_rewrite_blks) {
        all_blks.push_back(sp_iter);
      }
    }
    for (const Stmt& sp_iter : compute_blks) {
      all_blks.push_back(sp_iter);
    }
    Stmt new_body = all_blks.size() == 1 ? all_blks[0] : SeqStmt(all_blks);
    fptr->body = BlockRealize({}, const_true(), Block({}, {}, {}, "root", new_body, NullOpt, {}));
    return f;
  } else {
    return f;
  }
}

namespace transform {

Pass SparseFormatRewrite(Array<FormatRewriteRule> format_rewrite_rules,
                         bool include_format_rewrite_blks) {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return SparseFormatRewrite(std::move(format_rewrite_rules), std::move(f),
                               include_format_rewrite_blks);
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.SparseFormatRewrite", {});
}

TVM_REGISTER_GLOBAL("tir.transform.SparseFormatRewrite").set_body_typed(SparseFormatRewrite);

}  // namespace transform

}  // namespace tir
}  // namespace tvm