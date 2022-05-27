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

"""Specialize buffer with index map."""
from typing import Callable, Union
from tvm import IRModule
from tvm.tir import IndexMap
from tvm.tir.transform import SpecializeBuffer


def specialize_buffer(mod: IRModule, buf_name: str, idx_map: Union[Callable, IndexMap]):
    """Specialize a buffer in an IRModule with given buffer name and index map function.

    Parameters
    ----------
    mod : IRModule
        The IRModule we perform the specialization.

    buf_name : str
        The name of the buffer to specialize.

    idx_map : IndexMap
        The index map.

    Returns
    -------
    IRModule
        The new IRModule.
    """
    if isinstance(idx_map, Callable):
        idx_map = IndexMap.from_func(idx_map)
    return SpecializeBuffer(buf_name, idx_map)(mod)
