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
 * \file specialize_buffer.cc
 * \brief Specialize the element at given coordinate in a buffer with a specified index mapping.
 * e.g. replace buf[i, j, k] with index_map(i, j, k) in the program.
 */

#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <set>
#include <utility>

#include "../../support/utils.h"
#include "../schedule/analysis.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

class BufferSpecializer : public StmtExprMutator {
 public:
  explicit BufferSpecializer(const Buffer& buf, const IndexMap& idx_map)
      : buf_(buf), idx_map_(idx_map) {}

 private:
  Stmt VisitStmt_(const BufferStoreNode* op) final {
    CHECK(!op->buffer.same_as(buf_)) << "Cannot specialize a buffer to be written in the program.";
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    Array<PrimExpr> new_indices;
    for (const PrimExpr& idx : op->indices) {
      new_indices.push_back(ana_.Simplify(VisitExpr(idx)));
    }
    if (op->buffer.same_as(buf_)) {
      return ana_.Simplify(idx_map_->MapIndices(new_indices)[0]);
    } else {
      return BufferLoad(op->buffer, new_indices);
    }
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    auto n = CopyOnWrite(op);
    Array<BufferRegion> new_reads, new_writes;
    for (const BufferRegion& access : op->reads) {
      Buffer buf = access->buffer;
      Array<Range> new_region;
      for (const Range& range: access->region) {
        new_region.push_back(Range::FromMinExtent(
          ana_.Simplify(VisitExpr(range->min)), ana_.Simplify(VisitExpr(range->extent))
        ));
      }
      if (!buf.same_as(buf_)) {
        // remove read access to given buffer.
        new_reads.push_back(BufferRegion(buf, new_region));
      }
    }
    for (const BufferRegion& access : op->writes) {
      Buffer buf = access->buffer;
      Array<Range> new_region;
      for (const Range& range: access->region) {
        new_region.push_back(Range::FromMinExtent(
          ana_.Simplify(VisitExpr(range->min)), ana_.Simplify(VisitExpr(range->extent))
        ));
      }
      if (!buf.same_as(buf_)) {
        // remove write access to given buffer.
        new_writes.push_back(BufferRegion(buf, new_region));
      }
    }
    n->reads = new_reads;
    n->writes = new_writes;
    if (op->init.defined()) {
      n->init = VisitStmt(op->init.value());
    }
    n->body = VisitStmt(op->body);
    return Block(n);
  }

  const Buffer& buf_;
  IndexMap idx_map_;
  arith::Analyzer ana_;
};

PrimFunc SpecializeBuffer(const String& buf_name, const IndexMap& idx_map, PrimFunc f) {
  CHECK(idx_map->final_indices.size() == 1) << "The specified index map must have a single output.";
  // Only apply this pass to TIR that is not from TE schedules
  if (!IsFromLegacyTESchedule(f)) {
    PrimFuncNode* fptr = f.CopyOnWrite();
    Buffer buf;
    bool found = false;
    Map<Var, Buffer> new_buf_map;
    for (const auto& kv : f->buffer_map) {
      if (kv.second->name == buf_name) {
        buf = kv.second;
        found = true;
      } else {
        // Remove given buffer from buffer map.
        new_buf_map.Set(kv.first, kv.second);
      }
    }
    CHECK(found) << "Cannot find buffer with name " << buf_name;
    fptr->buffer_map = new_buf_map;
    fptr->body = BufferSpecializer(buf, idx_map)(std::move(fptr->body));
    return f;
  } else {
    return f;
  }
}

namespace transform {

Pass SpecializeBuffer(const String& buf_name, const IndexMap& idx_map) {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return SpecializeBuffer(buf_name, idx_map, std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.SpecializeBuffer", {});
}

TVM_REGISTER_GLOBAL("tir.transform.SpecializeBuffer").set_body_typed(SpecializeBuffer);

}  // namespace transform

}  // namespace tir
}  // namespace tvm