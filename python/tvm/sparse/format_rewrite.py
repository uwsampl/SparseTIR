# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Format search module for sparse tensor algebra."""
from typing import Callable, Dict, List
import tvm._ffi
import tvm.tir

from tvm.runtime import Object
from tvm.tir import IndexMap, _ffi_api


@tvm._ffi.register_object("tir.sparse.FormatRewriteRule")
class FormatRewriteRule(Object):
    """Format rewriting rule.

    Parameters
    ----------
    name : str
        Name of the format rewriting rule.
    format_desc : PrimFunc
        A TIR script describing the new format.
    buffers_to_rewrite: List[str]
        The list of sparse buffers we need to rewrite.
    axes_before_rewrite : List[str]
        The list of axes before the rewrite.
    axes_after_rewrite : List[str]
        The list of axes after the rewrite.
    axis_map : Dict[str, List[str]]
        The axis mapping from the old format to the new format.
    idx_map_func : Callable
        A function describing the index mapping from the old format to indices in new format.
    inv_idx_map_func : Callable
        A function describing the coordinate mapping from indices in new format.
        to indices in old format.
    """

    def __init__(
        self,
        name: str,
        new_format_desc: tvm.tir.PrimFunc,
        buffers_to_rewrite: List[str],
        axes_before_rewrite: List[str],
        axes_after_rewrite: List[str],
        axis_map: Dict[str, List[str]],
        idx_map_func: Callable,
        inv_idx_map_func: Callable,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.FormatRewriteRule,
            name,
            new_format_desc,
            buffers_to_rewrite,
            axes_before_rewrite,
            axes_after_rewrite,
            axis_map,
            IndexMap.from_func(idx_map_func),
            IndexMap.from_func(inv_idx_map_func),
        )  # type: ignore
