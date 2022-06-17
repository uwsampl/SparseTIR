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

void HideBufAccess(ScheduleState self, const StmtSRef& block_sref, const String& buf_type,
                   const Array<PrimExpr>& buf_index_array) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  Array<BufferRegion> reads, writes;
  std::set<int> buf_indices;
  for (const PrimExpr& buf_idx : buf_index_array) {
    const IntImmNode* int_imm = buf_idx.as<IntImmNode>();
    CHECK(int_imm != nullptr) << "buf_index_array must be a set of integers, got " << buf_idx
                              << " instead";
    buf_indices.insert(int_imm->value);
  }
  if (buf_type == "read") {
    for (size_t i = 0; i < block->reads.size(); ++i) {
      if (!buf_indices.count(i)) {
        reads.push_back(block->reads[i]);
      }
    }
    writes = block->writes;
  } else if (buf_type == "write") {
    for (size_t i = 0; i < block->writes.size(); ++i) {
      if (!buf_indices.count(i)) {
        writes.push_back(block->writes[i]);
      }
    }
    reads = block->reads;
  } else {
    LOG(INFO) << "Unregonized buffer type " << buf_type << ", only support read/write";
  }

  Block new_block(block->iter_vars, reads, writes, block->name_hint, block->body, block->init,
                  block->alloc_buffers, block->match_buffers, block->buf_doms, block->annotations);
  Map<Block, Block> blk_map;
  blk_map.Set(GetRef<Block>(block), new_block);
  self->Replace(block_sref, new_block, blk_map);
}

}  // namespace tir
}  // namespace tvm
