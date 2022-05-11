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
from typing import Callable
import tvm._ffi
import tvm.tir

from tvm.tir import IndexMap


class FormatRewriteRule:
    """Format rewriting rule.

    Parameters
    ----------
    name : str
        Name of the format rewriting rule.
    format_desc : PrimFunc
        A TIR script describing the new format.
    idx_map_func : Callable
        A function describing the coordinate mapping from indices in old format
        to indices in new format.
    inv_idx_map_func : Callable
        A function describing the coordinate mapping frmo indices in new format
        to indices in old format.
    params_transform_func : Callable
        A transformation function that turns the parameters (indptr/indices) of old format
        to parameters of new format.
    """

    def __init__(
        self,
        name: str,
        format_desc: tvm.tir.PrimFunc,
        idx_map_func: Callable,
        inv_idx_map_func: Callable,
        params_transform_func: Callable,
    ) -> None:
        self.name = name
        self.format_desc = format_desc
        self.idx_map = IndexMap.from_func(idx_map_func)
        self.inv_idx_map = IndexMap.from_func(inv_idx_map_func)
        self.params_transform = params_transform_func
