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

class SpIterationReplacer : public StmtMutator {
 public:
  explicit SpIterationReplacer(const SparseIteration& old_sp_iteration,
                               const SparseIteration& new_sp_iteration)
      : old_sp_iteration_(old_sp_iteration.get()), new_sp_iteration_(new_sp_iteration.get()) {}

 private:
  Stmt VisitStmt_(const SparseIterationNode* op) override {
    if (op == old_sp_iteration_) {
      return GetRef<SparseIteration>(new_sp_iteration_);
    } else {
      return StmtMutator::VisitStmt_(op);
    }
  }

  const SparseIterationNode *old_sp_iteration_, *new_sp_iteration_;
};

void UpdateIRModule(ScheduleState self, const SparseIteration& old_sp_iteration,
                    const SparseIteration& new_sp_iteration) {
  const PrimFuncNode* g_func = nullptr;
  GlobalVar g_var;
  g_func = GetPrimFuncFromSparseIteration(self->mod, old_sp_iteration.get(), &g_var);

  IRModuleNode* new_mod = self->mod.CopyOnWrite();
  MapNode* new_map = new_mod->functions.CopyOnWrite();
  PrimFunc ref_new_func = Downcast<PrimFunc>(std::move(new_map->at(g_var)));
  ICHECK(ref_new_func.get() == g_func);
  PrimFuncNode* new_func = ref_new_func.CopyOnWrite();

  SpIterationReplacer replacer(old_sp_iteration, new_sp_iteration);
  new_func->body = replacer(g_func->body);
  new_map->at(g_var) = std::move(ref_new_func);
  self->mod = GetRef<IRModule>(new_mod);
}

/*!
 * \brief Check whether the new iterators are valid. We say they are valid if the new order is a
 * permutation of the old order
 * \param new_order The new iterator order to be checked
 * \param old_order The old order of the iterators
 * \throw ScheduleError If the iterators in the new order are not valid
 */
void CheckValidInputIterators(const ScheduleState self, const Array<SpIterVar>& new_order,
                              const Array<SpIterVar>& old_order) {
  class LengthNotEqualError : public ScheduleError {
   public:
    explicit LengthNotEqualError(IRModule mod, Array<SpIterVar> old_order,
                                 Array<SpIterVar> new_order)
        : mod_(std::move(mod)), old_order_(std::move(old_order)), new_order_(std::move(new_order)) {
      ICHECK_NE(new_order_.size(), old_order_.size());
    }

    String FastErrorString() const final {
      return "ScheduleError: The number of iterators in the new order does not equal to the "
             "number of iterators in the old order";
    }

    String DetailRenderTemplate() const final {
      std::ostringstream os;
      os << "ScheduleError: The new order has " << new_order_.size() << " iterators" << new_order_
         << ", while the old order has " << old_order_.size() << " iterators" << old_order_
         << ". They are supposed to have the same set of iterators";
      return os.str();
    }

    IRModule mod() const final { return mod_; }
    Array<ObjectRef> LocationsOfInterest() const final { return {}; }

    IRModule mod_;
    Array<SpIterVar> old_order_;
    Array<SpIterVar> new_order_;
  };

  class IterNotAppearError : public ScheduleError {
   public:
    explicit IterNotAppearError(IRModule mod, SpIterVar iter, Array<SpIterVar> old_order)
        : mod_(std::move(mod)), iter_(std::move(iter)), old_order_(std::move(old_order)) {}

    String FastErrorString() const final {
      return "ScheduleError: An iterator in the new order does not appear in the old order";
    }

    String DetailRenderTemplate() const final {
      std::ostringstream os;
      os << "ScheduleError: Iterator " << iter_
         << " appears in the new order. However, it does not appear in the old order "
         << old_order_;
      return os.str();
    }

    IRModule mod() const final { return mod_; }
    Array<ObjectRef> LocationsOfInterest() const final { return {}; }

    IRModule mod_;
    SpIterVar iter_;
    Array<SpIterVar> old_order_;
  };

  if (new_order.size() != old_order.size()) {
    throw LengthNotEqualError(self->mod, new_order, old_order);
  }
  for (const SpIterVar& sp_iter : new_order) {
    if (std::find(old_order.begin(), old_order.end(), sp_iter) == old_order.end()) {
      throw IterNotAppearError(self->mod, sp_iter, old_order);
    }
  }
}

/*!
 * \brief Check whether the sparse reorder would break dependency between iterators.
 * \param new_order The new iterator order to be checked.
 * \throw ScheduleError If the sparse reorder breaks dependency.
 */
void CheckDependency(const ScheduleState self, const Array<SpIterVar>& new_order) {
  class DependencyError : public ScheduleError {
   public:
    explicit DependencyError(IRModule mod, SpIterVar iter, Array<SpIterVar> new_order)
        : mod_(std::move(mod)), iter_(std::move(iter)), new_order_(std::move(new_order)) {}

    String FastErrorString() const final {
      return "ScheduleError: the sparse reorder breaks dependency between axes.";
    }

    String DetailRenderTemplate() const final {
      std::ostringstream os;
      os << "ScheduleError: in new order " << new_order_ << " iterator " << iter_
         << " was placed before its dependent iterator.";
      return os.str();
    }

    IRModule mod() const final { return mod_; }
    Array<ObjectRef> LocationsOfInterest() const final { return {}; }

    IRModule mod_;
    SpIterVar iter_;
    Array<SpIterVar> new_order_;
  };

  std::set<Axis> axes_set;
  for (const SpIterVar& sp_iter : new_order) {
    Axis axis = sp_iter->axis;
    auto try_parent = axis->parent;
    if (try_parent.defined()) {
      Axis parent = Downcast<Axis>(try_parent.value());
      if (axes_set.find(parent) == axes_set.end()) {
        throw DependencyError(self->mod, sp_iter, new_order);
      }
    }
    axes_set.insert(axis);
  }
}

SparseIteration SparseReorder(ScheduleState self, const SparseIteration& sp_iteration,
                              const Array<SpIterVar>& new_order) {
  // Step 1. Check whether the iterators in `new_order` are the same as `sp_iteration`'s iterators.
  CheckValidInputIterators(self, new_order, sp_iteration->sp_iter_vars);

  // Step 2. Check whether the new order does not break the iterator dependency.
  CheckDependency(self, new_order);

  // Step 3. Create the new SparseIteration.
  ObjectPtr<SparseIterationNode> p_new_sp_iteration =
      make_object<SparseIterationNode>(*sp_iteration.get());
  p_new_sp_iteration->sp_iter_vars = new_order;
  SparseIteration new_sp_iteration(p_new_sp_iteration);

  UpdateIRModule(self, sp_iteration, new_sp_iteration);
  return new_sp_iteration;
}

int CheckFuseMatch(const ScheduleState self, const Array<SpIterVar>& iters_to_fuse,
                   const Array<SpIterVar>& blk_iter_vars) {
  class FuseNotMatchError : public ScheduleError {
   public:
    explicit FuseNotMatchError(IRModule mod, const Array<SpIterVar>& iters_to_fuse,
                               const Array<SpIterVar>& blk_iter_vars)
        : mod_(std::move(mod)),
          iters_to_fuse_(std::move(iters_to_fuse)),
          blk_iter_vars_(std::move(blk_iter_vars)) {}

    String FastErrorString() const final {
      return "ScheduleError: The sparse iter vars to fuse doesn't match the sparse iter vars in "
             "the given sparse iteration.";
    }

    String DetailRenderTemplate() const final {
      std::ostringstream os;
      os << "ScheduleError: The sparse iter vars to fuse: " << iters_to_fuse_ << " doesn't match "
         << "the sparse iter vars in the given sparse iteration: " << blk_iter_vars_;
      return os.str();
    }

    IRModule mod() const final { return mod_; }
    Array<ObjectRef> LocationsOfInterest() const final { return {}; }

    IRModule mod_;
    Array<SpIterVar> iters_to_fuse_;
    Array<SpIterVar> blk_iter_vars_;
  };

  size_t j = 0;
  size_t match_pos = 0;
  for (size_t i = 0; i < blk_iter_vars.size(); ++i) {
    if (blk_iter_vars[i].same_as(iters_to_fuse[j])) {
      if (j == 0) {
        match_pos = i;
      }
      j++;
      if (j == iters_to_fuse.size()) {
        return match_pos;
      }
    }
  }
  throw FuseNotMatchError(self->mod, iters_to_fuse, blk_iter_vars);
  return -1;
}

SparseIteration SparseFuse(ScheduleState self, const SparseIteration& sp_iteration,
                           const Array<SpIterVar>& iters_to_fuse) {
  // Step 1. Check match or not.
  int match_pos = CheckFuseMatch(self, iters_to_fuse, sp_iteration->sp_iter_vars);

  ObjectPtr<SparseIterationNode> p_new_sp_iteration =
      make_object<SparseIterationNode>(*sp_iteration.get());
  Array<SpIterVar> new_sp_iters;
  for (int i = 0; i < match_pos; ++i) {
    new_sp_iters.push_back(sp_iteration->sp_iter_vars[i]);
  }
  Array<Axis> axis_group;
  for (const SpIterVar& sp_iter_var : iters_to_fuse) {
    axis_group.push_back(sp_iter_var->axis);
  }
  for (size_t i = 0; i < iters_to_fuse.size(); ++i) {
    const SpIterVar& sp_iter_var = iters_to_fuse[i];
    Axis new_axis = FusedAxis(axis_group, i);
    new_sp_iters.push_back(SpIterVar(sp_iter_var->var, sp_iter_var->is_reduction, new_axis));
  }
  for (size_t i = match_pos + iters_to_fuse.size(); i < sp_iteration->sp_iter_vars.size(); ++i) {
    new_sp_iters.push_back(sp_iteration->sp_iter_vars[i]);
  }
  p_new_sp_iteration->sp_iter_vars = new_sp_iters;
  SparseIteration new_sp_iteration(p_new_sp_iteration);

  UpdateIRModule(self, sp_iteration, new_sp_iteration);
  return new_sp_iteration;
}

SparseIteration GetSparseIteration(const ScheduleState& self, const String& name,
                                   const String& func_name) {
  class Finder : public StmtVisitor {
   public:
    explicit Finder(const ScheduleState& self, const String& name) : self_(self), name_(name) {}

    void VisitStmt_(const SparseIterationNode* op) override {
      if (op->name == name_) {
        result_ = GetRef<SparseIteration>(op);
      }
      StmtVisitor::VisitStmt_(op);
    }

    const ScheduleState& self_;
    const String& name_;
    Optional<SparseIteration> result_;
  };

  BaseFunc func = self->mod->Lookup(func_name);
  const auto* prim_func = TVM_TYPE_AS(prim_func, func, PrimFuncNode);

  Finder finder(self, name);
  finder(prim_func->body);
  CHECK(finder.result_.defined()) << "Cannot find a sparse iteration with the name: " + name;
  return finder.result_.value();
}

SparseIteration AnnotateSparseIteration(ScheduleState self, const SparseIteration& sp_iteration,
                                        const String& ann_key, const ObjectRef& ann_val) {
  ObjectPtr<SparseIterationNode> p_new_sp_iteration =
      make_object<SparseIterationNode>(*sp_iteration.get());
  p_new_sp_iteration->annotations.Set(ann_key, ann_val);
  SparseIteration new_sp_iteration(p_new_sp_iteration);
  UpdateIRModule(self, sp_iteration, new_sp_iteration);
  return new_sp_iteration;
}

SparseIteration UnannotateSparseIteration(ScheduleState self, const SparseIteration& sp_iteration,
                                          const String& ann_key) {
  ObjectPtr<SparseIterationNode> p_new_sp_iteration =
      make_object<SparseIterationNode>(*sp_iteration.get());
  p_new_sp_iteration->annotations.erase(ann_key);
  SparseIteration new_sp_iteration(p_new_sp_iteration);
  UpdateIRModule(self, sp_iteration, new_sp_iteration);
  return new_sp_iteration;
}

}  // namespace tir
}  // namespace tvm
