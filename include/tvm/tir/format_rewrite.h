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
 * \brief tvm/tir/format_rewrite.h
 * \brief .
 */
#ifndef TVM_TIR_FORMAT_REWRITE_H_
#define TVM_TIR_FORMAT_REWRITE_H_

#include <tvm/tir/function.h>
#include <tvm/tir/index_map.h>

namespace tvm {

namespace tir {

class FormatRewriteRuleNode : public Object {
 public:
  String name;
  PrimFunc new_format_desc;
  Array<String> buffers_to_rewrite;
  Array<String> axes_before_rewrite;
  Array<String> axes_after_rewrite;
  Map<String, Array<String>> axis_map;
  IndexMap idx_map;
  IndexMap inv_idx_map;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("new_format_desc", &new_format_desc);
    v->Visit("buffers_to_rewrite", &buffers_to_rewrite);
    v->Visit("axes_before_rewrite", &axes_before_rewrite);
    v->Visit("axes_after_rewrite", &axes_after_rewrite);
    v->Visit("axis_map", &axis_map);
    v->Visit("idx_map", &idx_map);
    v->Visit("inv_idx_map", &inv_idx_map);
  }

  static constexpr const char* _type_key = "tir.sparse.FormatRewriteRule";
  TVM_DECLARE_FINAL_OBJECT_INFO(FormatRewriteRuleNode, Object);
};

class FormatRewriteRule : public ObjectRef {
 public:
  TVM_DLL explicit FormatRewriteRule(String name, PrimFunc new_format_desc,
                                     Array<String> buffers_to_rewrite,
                                     Array<String> axes_before_rewrite,
                                     Array<String> axes_after_rewrite,
                                     Map<String, Array<String>> axis_map, IndexMap idx_map,
                                     IndexMap inv_idx_map);

  TVM_DEFINE_OBJECT_REF_METHODS(FormatRewriteRule, ObjectRef, FormatRewriteRuleNode);
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_FORMAT_REWRITE_H_
