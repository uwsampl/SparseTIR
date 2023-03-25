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

"""Lower Sparse Iterators and Lower Sparse Buffers for Sparse TIR."""
from tvm import IRModule
from tvm.tir.transform import LowerSparseBuffer, LowerSparseIter


def lower_sparse_iter(mod: IRModule, check_invalid_binary_search: bool = False):
    """Lower sparse iterators in Sparse TIR.

    Parameters
    ----------
    mod : IRModule
        The IRModule to lower.
    check_invalid_binary_search : bool
        Whether check invalid indices made by binary search.
    """
    if not isinstance(mod, IRModule):
        raise TypeError("Expected IRModule, but got {}".format(type(mod)))
    return LowerSparseIter(check_invalid_binary_search)(mod)


def lower_sparse_buffer(mod: IRModule):
    """Lower sparse buffers in Sparse TIR.

    Parameters
    ----------
    mod : IRModule
        The IRModule to lower.
    """
    if not isinstance(mod, IRModule):
        raise TypeError("Expected IRModule, but got {}".format(type(mod)))
    return LowerSparseBuffer()(mod)
