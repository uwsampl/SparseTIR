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
#include "../../ir/functor_common.h"
#include "../utils.h"

namespace tvm {
namespace tir {

class StorageAlignAxisOutOfRangeError : public ScheduleError {
 public:
  explicit StorageAlignAxisOutOfRangeError(IRModule mod, Buffer buffer, int axis)
      : mod_(std::move(mod)), buffer_(std::move(buffer)), axis_(axis) {}

  String FastErrorString() const final {
    return "ScheduleError: The input `axis` is out of range. It is required to be in range "
           "[-ndim, ndim) where `ndim` is the number of dimensions of the buffer to set "
           "storage alignment.";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    int ndim = static_cast<int>(buffer_->shape.size());
    os << "The buffer to set storage alignment of, " << buffer_->name << ", has " << ndim
       << " dimension(s), so `axis` is required to be in [" << -(ndim) << ", " << ndim
       << ") for storage_align. However, the input `axis` is " << axis_
       << ", which is out of the expected range.";
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {}; }

  static int CheckAndUpdate(const IRModule& mod, const Buffer& buffer, int axis) {
    int ndim = static_cast<int>(buffer->shape.size());
    if (axis < -ndim || axis >= ndim) {
      throw StorageAlignAxisOutOfRangeError(mod, buffer, axis);
    }
    // If axis is negative, convert it to a non-negative one.
    if (axis < 0) {
      axis += ndim;
    }
    return axis;
  }

 private:
  IRModule mod_;
  Buffer buffer_;
  int axis_;
};

class NonAllocatedBufferError : public ScheduleError {
 public:
  explicit NonAllocatedBufferError(IRModule mod, Buffer buffer) : mod_(mod), buffer_(buffer) {}

  String FastErrorString() const final {
    return "ScheduleError: The input buffer is not allocated by a block. This means the buffer is "
           " either a function parameter or defined in `match_buffer` of a block.";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The input buffer " << buffer_->name
       << " is not allocated by a block. This means the buffer is either a function parameter or "
          "defined in `match_buffer` of a block.";
    return os.str();
  }

  static StmtSRef CheckAndGetBufferAllocationSite(const IRModule& mod, const StmtSRef& block_sref,
                                                  const Buffer& buffer) {
    Optional<StmtSRef> defining_site_sref;
    bool is_alloc;
    std::tie(defining_site_sref, is_alloc) = GetBufferDefiningSite(block_sref, buffer);
    if (!defining_site_sref.defined() || !is_alloc) {
      throw NonAllocatedBufferError(mod, buffer);
    }

    return defining_site_sref.value();
  }

  Array<ObjectRef> LocationsOfInterest() const final { return {}; }
  IRModule mod() const final { return mod_; }

 private:
  IRModule mod_;
  Buffer buffer_;
};

class StorageAlignInvalidFactorError : public ScheduleError {
 public:
  explicit StorageAlignInvalidFactorError(IRModule mod, int factor)
      : mod_(std::move(mod)), factor_(factor) {}

  String FastErrorString() const final {
    return "ScheduleError: The input `factor` of storage_align is expected to be a positive "
           "number.";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The input `factor` of storage_align is expected to be a positive number. However, the "
          "input `factor` is "
       << factor_ << ", which is out of the expected range.";
    return os.str();
  }

  static void Check(const IRModule& mod, int factor) {
    if (factor <= 0) {
      throw StorageAlignInvalidFactorError(mod, factor);
    }
  }

  Array<ObjectRef> LocationsOfInterest() const final { return {}; }
  IRModule mod() const final { return mod_; }

 private:
  IRModule mod_;
  int factor_;
};

class StorageAlignInvalidAnnotationError : public ScheduleError {
 public:
  explicit StorageAlignInvalidAnnotationError(IRModule mod, Block block)
      : mod_(std::move(mod)), block_(std::move(block)) {}

  String FastErrorString() const final {
    return "ScheduleError: The block annotation for storage align is expected to be an array of "
           "4-integer-tuples (buffer_index, axis, factor, offset).";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The block annotation for storage align is expected to be an array of 4-integer-tuples "
          "(buffer_index, axis, factor, offset). However, the block annotation with key "
       << attr::buffer_dim_align << " of the block {0} is "
       << block_->annotations.at(attr::buffer_dim_align) << ", which is unexpected.";
    return os.str();
  }

  static StorageAlignAnnotation CheckAndGetAnnotation(const IRModule& mod, const Block& block) {
    // Get existing annotation value.
    auto it = block->annotations.find(attr::buffer_dim_align);
    if (it != block->annotations.end()) {
      if (!IsValidAnnotation(block, (*it).second)) {
        throw StorageAlignInvalidAnnotationError(mod, block);
      }
      return Downcast<StorageAlignAnnotation>((*it).second);
    }

    // Create new annotation value
    StorageAlignAnnotation storage_align_annotation;
    return storage_align_annotation;
  }

  Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }
  IRModule mod() const final { return mod_; }

 private:
  static bool IsValidAnnotation(const Block& block, const ObjectRef& anno_value) {
    if (!anno_value->IsInstance<ArrayNode>()) {
      return false;
    }
    auto storage_align_annotations = Downcast<Array<ObjectRef>>(anno_value);
    for (const ObjectRef& storage_align_annotation : storage_align_annotations) {
      if (!storage_align_annotation->IsInstance<ArrayNode>()) {
        return false;
      }
      auto storage_align_tuple = Downcast<Array<ObjectRef>>(storage_align_annotation);
      // Check if the annotation is a 4-tuple.
      if (storage_align_tuple.size() != 4) {
        return false;
      }
      for (const ObjectRef& tuple_element : storage_align_tuple) {
        if (!tuple_element->IsInstance<IntImmNode>()) {
          return false;
        }
      }
    }
    return true;
  }

  IRModule mod_;
  Block block_;
};

class MatchToAllocMutator : StmtExprMutator {
 public:
  /*!
   * \param root_block The root block to rewrite.
   * \param old_buffer The old buffer.
   * \param new_buffer The new allocated buffer.
   * \param block_sref_reuse The block sref reuse map to be updated.
   */
  static Block Mutate(const Block& root_block, const Buffer& old_buffer, const Buffer& new_buffer,
                      Map<Block, Block>* block_sref_reuse) {
    MatchToAllocMutator mutator(old_buffer, new_buffer, block_sref_reuse);
    Stmt new_block = mutator.VisitStmt(root_block);
    return Downcast<Block>(new_block);
  }

 private:
  MatchToAllocMutator(const Buffer& old_buffer, const Buffer& new_buffer,
                      Map<Block, Block>* block_sref_reuse)
      : old_buffer_(old_buffer), new_buffer_(new_buffer), block_sref_reuse_(block_sref_reuse) {}

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    if (op->buffer.same_as(old_buffer_)) {
      Array<PrimExpr> indices;
      for (const PrimExpr& idx : op->indices) {
        indices.push_back(StmtExprMutator::VisitExpr(idx));
      }
      return BufferLoad(new_buffer_, indices);
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    if (op->buffer.same_as(old_buffer_)) {
      Array<PrimExpr> indices;
      for (const PrimExpr& idx : op->indices) {
        indices.push_back(StmtExprMutator::VisitExpr(idx));
      }
      return BufferStore(new_buffer_, StmtExprMutator::VisitExpr(op->value), indices);
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    Block old_block = GetRef<Block>(op);
    Block mutated_block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    ObjectPtr<BlockNode> n = CopyOnWrite(mutated_block.get());

    if (mutated_block->name_hint == "root") {
      n->alloc_buffers.push_back(new_buffer_);
    } else {
      auto f_mutate_read_write_region = [this](const BufferRegion& buffer_region) {
        return buffer_region->buffer.same_as(old_buffer_)
                   ? BufferRegion(new_buffer_, buffer_region->region)
                   : buffer_region;
      };

      // Step 2. Mutate the read/write region.
      Array<BufferRegion> reads = MutateArray(n->reads, f_mutate_read_write_region);
      Array<BufferRegion> writes = MutateArray(n->writes, f_mutate_read_write_region);

      n->reads = reads;
      n->writes = writes;
    }
    Block new_block(n);
    block_sref_reuse_->Set(old_block, new_block);
    return new_block;
  }

  Buffer old_buffer_;
  Buffer new_buffer_;
  Map<Block, Block>* block_sref_reuse_;
};

/*!
 * \brief A helper mutator which recursively mutates the old buffer's storage scope and collects
 * the block sref reuse information for the following replacement.
 */
class StorageScopeMutator : StmtExprMutator {
 public:
  /*!
   * \param allocate_site The block where `old_buffer` was allocated.
   * \param old_buffer The old buffer
   * \param storage_scope The storage scope to be set
   * \param block_sref_reuse The block sref reuse map to be updated
   * \return The new block after the mutation
   */
  static Block Mutate(const Block& allocate_site, const Buffer& old_buffer,
                      const String& storage_scope, Map<Block, Block>* block_sref_reuse) {
    Buffer new_buffer = WithScope(old_buffer, storage_scope);
    StorageScopeMutator mutator(old_buffer, new_buffer, storage_scope, block_sref_reuse);
    Stmt new_block = mutator.VisitStmt(allocate_site);
    return Downcast<Block>(new_block);
  }

 private:
  StorageScopeMutator(const Buffer& old_buffer, Buffer new_buffer, String storage_scope,
                      Map<Block, Block>* block_sref_reuse)
      : storage_scope_(std::move(storage_scope)), block_sref_reuse_(block_sref_reuse) {
    buffer_var_map_[old_buffer->data.get()] = std::move(new_buffer);
  }

  PrimExpr VisitExpr_(const VarNode* var) final {
    auto it = buffer_var_map_.find(var);
    return it != buffer_var_map_.end() ? it->second->data : GetRef<Var>(var);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* load) final {
    BufferLoad res = Downcast<BufferLoad>(ExprMutator::VisitExpr_(load));

    auto it = buffer_var_map_.find(res->buffer->data.get());
    if (it != buffer_var_map_.end()) {
      ObjectPtr<BufferLoadNode> ptr = make_object<BufferLoadNode>(*res.get());
      ptr->buffer = it->second;
      return PrimExpr(ptr);
    } else {
      return std::move(res);
    }
  }

  Stmt VisitStmt_(const BufferStoreNode* store) final {
    BufferStore res = Downcast<BufferStore>(StmtMutator::VisitStmt_(store));

    auto it = buffer_var_map_.find(res->buffer->data.get());
    if (it != buffer_var_map_.end()) {
      ObjectPtr<BufferStoreNode> ptr = make_object<BufferStoreNode>(*res.get());
      ptr->buffer = it->second;
      return Stmt(ptr);
    } else {
      return std::move(res);
    }
  }

  Stmt VisitStmt_(const BlockNode* block) final {
    // To reduce the number of blocks in block sref reuse map, we check whether the block is really
    // mutated (i.e., the old buffer appears in the block). If so, we return the block after
    // mutation. Otherwise we just return the original block.

    // Define the mutation functions.
    auto f_mutate_match_buffers = [this](const MatchBufferRegion& match_buffer) {
      auto it = buffer_var_map_.find(match_buffer->source->buffer->data.get());
      if (it != buffer_var_map_.end()) {
        Buffer new_target_buffer = WithScope(match_buffer->buffer, storage_scope_);
        buffer_var_map_[match_buffer->buffer->data.get()] = new_target_buffer;
        return MatchBufferRegion(new_target_buffer,
                                 BufferRegion(it->second, match_buffer->source->region));
      } else {
        return match_buffer;
      }
    };
    auto f_mutate_read_write_region = [this](const BufferRegion& buffer_region) {
      auto it = buffer_var_map_.find(buffer_region->buffer->data.get());
      return it == buffer_var_map_.end() ? buffer_region
                                         : BufferRegion(it->second, buffer_region->region);
    };
    auto f_mutate_alloc_buffers = [this](const Buffer& buffer) {
      auto it = buffer_var_map_.find(buffer->data.get());
      return it == buffer_var_map_.end() ? buffer : it->second;
    };

    // Step 1. Mutate `match_buffers`. If an old buffer appears as a source of MatchBufferRegion,
    // the storage scope of the target buffer also needs to be set.
    Array<MatchBufferRegion> match_buffers =
        MutateArray(block->match_buffers, f_mutate_match_buffers);
    // Step 2. Mutate the read/write region.
    Array<BufferRegion> reads = MutateArray(block->reads, f_mutate_read_write_region);
    Array<BufferRegion> writes = MutateArray(block->writes, f_mutate_read_write_region);
    // Step 3. Mutate `alloc_buffers` for the old buffer allocated in this block.
    Array<Buffer> alloc_buffers = MutateArray(block->alloc_buffers, f_mutate_alloc_buffers);
    // Step 4. Recursively mutate the block.
    Block mutated_block = Downcast<Block>(StmtMutator::VisitStmt_(block));

    if (mutated_block.get() == block && reads.same_as(mutated_block->reads) &&
        writes.same_as(mutated_block->writes) &&
        alloc_buffers.same_as(mutated_block->alloc_buffers) &&
        match_buffers.same_as(mutated_block->match_buffers)) {
      return GetRef<Block>(block);
    } else {
      ObjectPtr<BlockNode> n = CopyOnWrite(mutated_block.get());
      n->reads = std::move(reads);
      n->writes = std::move(writes);
      n->alloc_buffers = std::move(alloc_buffers);
      n->match_buffers = std::move(match_buffers);

      Block new_block(n);
      block_sref_reuse_->Set(GetRef<Block>(block), new_block);
      return std::move(new_block);
    }
  }

  /*! \brief The storage scope to be set. */
  String storage_scope_;
  /*!
   * \brief A mapping which maps old buffer vars to new buffers, including the buffers defined in
   * MatchBufferRegion.
   */
  std::unordered_map<const VarNode*, Buffer> buffer_var_map_;
  /*! \brief The block sref reuse map for the following replacement */
  Map<Block, Block>* block_sref_reuse_;
};

void StorageAlign(ScheduleState self, const StmtSRef& block_sref, int buffer_index, int axis,
                  int factor, int offset) {
  const BlockNode* block_ptr = TVM_SREF_TO_BLOCK(block_ptr, block_sref);
  Buffer buffer =
      GetNthAccessBuffer(self, GetRef<Block>(block_ptr), buffer_index, /*is_write=*/true);
  StorageAlignInvalidFactorError::Check(self->mod, factor);
  axis = StorageAlignAxisOutOfRangeError::CheckAndUpdate(self->mod, buffer, axis);
  NonAllocatedBufferError::CheckAndGetBufferAllocationSite(self->mod, block_sref, buffer);

  // Step 1: Get existing or create new annotation value.
  StorageAlignAnnotation storage_align_annotation =
      StorageAlignInvalidAnnotationError::CheckAndGetAnnotation(self->mod,
                                                                GetRef<Block>(block_ptr));

  // Step 2: Update the annotation value
  bool found = false;
  StorageAlignTuple new_storage_align_tuple{Integer(buffer_index), Integer(axis), Integer(factor),
                                            Integer(offset)};
  for (size_t j = 0; j < storage_align_annotation.size(); ++j) {
    const auto& storage_align_tuple = storage_align_annotation[j];
    ICHECK(storage_align_tuple.size() == 4);
    if (storage_align_tuple[0] == buffer_index && storage_align_tuple[1] == axis) {
      storage_align_annotation.Set(j, std::move(new_storage_align_tuple));
      found = true;
      break;
    }
  }
  if (!found) {
    storage_align_annotation.push_back(std::move(new_storage_align_tuple));
  }

  // Step 3: Replace the block with the new annotation
  Block new_block = WithAnnotation(block_ptr, attr::buffer_dim_align, storage_align_annotation);
  self->Replace(block_sref, new_block, {{GetRef<Block>(block_ptr), new_block}});
}

void SetScope(ScheduleState self, const StmtSRef& block_sref, int buffer_index,
              const String& storage_scope) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  Buffer buffer = GetNthAccessBuffer(self, GetRef<Block>(block), buffer_index, true);

  // Step 1. If `storage_scope` equals the original storage scope of the buffer, just return.
  if (buffer.scope() == storage_scope) {
    return;
  }

  // Step 2. Throw an error if the input storage scope is invalid.
  CheckStorageScope(self, storage_scope);

  // Step 3. Get the allocation site of the target buffer.
  StmtSRef alloc_site_sref =
      NonAllocatedBufferError::CheckAndGetBufferAllocationSite(self->mod, block_sref, buffer);
  const BlockNode* alloc_site = TVM_SREF_TO_BLOCK(alloc_site, alloc_site_sref);

  // Step 4. Recursively replace the old buffer to a new buffer, where the new buffer has the given
  // storage scope. In the meanwhile, collect the block sref reuse information.
  Map<Block, Block> block_reuse_map;
  Block new_block = StorageScopeMutator::Mutate(GetRef<Block>(alloc_site), buffer, storage_scope,
                                                &block_reuse_map);
  self->Replace(alloc_site_sref, new_block, block_reuse_map);
}

void MatchToAlloc(ScheduleState self, const StmtSRef& block_sref, int buffer_index) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  Buffer buffer = GetNthAccessBuffer(self, GetRef<Block>(block), buffer_index, true);

  // Check whether the buffer is a top-level matched buffer.
  Optional<StmtSRef> defining_site_sref;
  bool is_alloc;
  std::tie(defining_site_sref, is_alloc) = GetBufferDefiningSite(block_sref, buffer);
  CHECK(!defining_site_sref.defined() && !is_alloc)
      << "The buffer to transform must be a top-level matched buffer.";

  // Create new allocated buffer.
  ObjectPtr<BufferNode> alloc_buffer_node = make_object<BufferNode>(*buffer.get());
  alloc_buffer_node->name = buffer->name + "_tmp";
  Buffer alloc_buffer = Buffer(alloc_buffer_node);

  const StmtSRefNode* root_block_sref_node = nullptr;
  const BlockNode* root_block_node = nullptr;
  for (const StmtSRefNode* p = block_sref.get(); p; p = p->parent) {
    if (p->stmt->IsInstance<BlockNode>()) {
      const BlockNode* block = TVM_SREF_TO_BLOCK(block, GetRef<StmtSRef>(p));
      if (block->name_hint == "root") {
        root_block_node = block;
        root_block_sref_node = p;
      }
    }
  }
  // Find root block and root block sref.
  Map<Block, Block> block_reuse_map;
  Block new_root_block = MatchToAllocMutator::Mutate(GetRef<Block>(root_block_node), buffer,
                                                     alloc_buffer, &block_reuse_map);

  self->Replace(GetRef<StmtSRef>(root_block_sref_node), new_root_block, block_reuse_map);
}

/******** InstructionKind Registration ********/

struct StorageAlignTraits : public UnpackedInstTraits<StorageAlignTraits> {
  static constexpr const char* kName = "StorageAlign";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 4;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV block_rv, Integer buffer_index,
                                      Integer axis, Integer factor, Integer offset) {
    return sch->StorageAlign(block_rv, buffer_index->value, axis->value, factor->value,
                             offset->value);
  }

  static String UnpackedAsPython(Array<String> outputs, String block_rv, Integer buffer_index,
                                 Integer axis, Integer factor, Integer offset) {
    PythonAPICall py("storage_align");
    py.Input("block", block_rv);
    py.Input("buffer_index", buffer_index);
    py.Input("axis", axis);
    py.Input("factor", factor);
    py.Input("offset", offset);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct SetScopeTraits : public UnpackedInstTraits<SetScopeTraits> {
  static constexpr const char* kName = "SetScope";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV block_rv, Integer buffer_index,
                                      String storage_scope) {
    return sch->SetScope(block_rv, buffer_index->value, storage_scope);
  }

  static String UnpackedAsPython(Array<String> outputs, String block_rv, Integer buffer_index,
                                 String storage_scope) {
    PythonAPICall py("set_scope");
    py.Input("block", block_rv);
    py.Input("buffer_index", buffer_index);
    py.Input("storage_scope", storage_scope);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct MatchToAllocTraits : public UnpackedInstTraits<MatchToAllocTraits> {
  static constexpr const char* kName = "MatchToAlloc";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV block_rv, Integer buffer_index) {
    return sch->MatchToAlloc(block_rv, buffer_index->value);
  }

  static String UnpackedAsPython(Array<String> outputs, String block_rv, Integer buffer_index) {
    PythonAPICall py("match_to_alloc");
    py.Input("block", block_rv);
    py.Input("buffer_index", buffer_index);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(StorageAlignTraits);
TVM_REGISTER_INST_KIND_TRAITS(SetScopeTraits);
TVM_REGISTER_INST_KIND_TRAITS(MatchToAllocTraits);

}  // namespace tir
}  // namespace tvm
