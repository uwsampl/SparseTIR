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

/******** Error Classes ********/

/******** Helper Functions/Classes ********/

/*! \brief Mutator for ReverseCacheRead */
class ReverseCacheReadRewriter : public StmtExprMutator {};

/*! \brief Mutator for ReverseCacheWrite */
class ReverseCacheWriteRewriter : public StmtExprMutator {};

/******** Implementation ********/

StmtSRef ReverseCacheRead(ScheduleState self, const StmtSRef& block_sref, int read_buffer_index,
                          const String& storage_scope) {}

StmtSRef ReverseCacheWrite(ScheduleState self, const StmtSRef& block_sref, int write_buffer_index,
                           const String& storage_scope) {}

/******** Instruction Registration ********/

struct ReverseCacheReadTraits : public UnpackedInstTraits<ReverseCacheReadTraits> {
  static constexpr const char* kName = "ReverseCacheRead";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static BlockRV UnpackedApplyToSchedule(Schedule sch, BlockRV block, Integer read_buffer_index,
                                         String storage_scope) {
    return sch->ReverseCacheRead(block, read_buffer_index->value, storage_scope);
  }

  static String UnpackedAsPython(Array<String> outputs, String block, Integer read_buffer_index,
                                 String storage_scope) {
    PythonAPICall py("reverse_cache_read");
    py.Input("block", block);
    py.Input("read_buffer_index", read_buffer_index->value);
    py.Input("storage_scope", storage_scope);
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct ReverseCacheWriteTraits : public UnpackedInstTraits<ReverseCacheWriteTraits> {
  static constexpr const char* kName = "ReverseCacheWrite";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static BlockRV UnpackedApplyToSchedule(Schedule sch, BlockRV block, Integer write_buffer_index,
                                         String storage_scope) {
    return sch->ReverseCacheWrite(block, write_buffer_index->value, storage_scope);
  }

  static String UnpackedAsPython(Array<String> outputs, String block, Integer write_buffer_index,
                                 String storage_scope) {
    PythonAPICall py("reverse_cache_write");
    py.Input("block", block);
    py.Input("write_buffer_index", write_buffer_index->value);
    py.Input("storage_scope", storage_scope);
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(ReverseCacheReadTraits);
TVM_REGISTER_INST_KIND_TRAITS(ReverseCacheWriteTraits);

}  // namespace tir
}  // namespace tvm
