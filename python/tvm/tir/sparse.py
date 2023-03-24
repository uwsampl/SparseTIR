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
"""SparseTIR related data structures"""
from typing import List, Optional
from enum import Enum

import tvm._ffi
from tvm.ir import PrimExpr
from tvm.ir.base import Span
from tvm.runtime import Object
from tvm.tir import Var

from . import _ffi_api
from .buffer import Buffer


class AxisKind(Enum):
    DenseFixed = 0
    DenseVariable = 1
    SparseFixed = 2
    SparseVariable = 3


@tvm._ffi.register_object("tir.sparse.Axis")
class Axis(Object):
    """Base class of all the sparse axes.

    Parameters
    ----------
    name : str
        Name of the axis.

    parent : Optional[Axis]
        The parent axis.

    length : PrimExpr
        The length upperbound of current axis.

    nnz : PrimExpr
        The accumulated number of nonzero elements from root axis to current axis.

    nnz_cols : Optional[PrimExpr]
        The number of nonzero columns in current row, only valid for fixed axis.

    indptr : Optional[Var]
        The indptr buffer var.

    indices : Optional[Var]
        The indices buffer var.

    idtype : str
        The index data type.

    sorted : bool
        The indices are sorted or not.
    """

    name: str
    parent: Optional["Axis"]
    length: PrimExpr
    nnz: PrimExpr
    nnz_cols: Optional[PrimExpr]
    indptr: Optional[Var]
    indices: Optional[Var]
    idtype: str
    sorted: bool

    def __init__(
        self, name, parent, length, nnz, nnz_cols, indptr, indices, idtype, sorted
    ) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Axis, name, parent, length, nnz, nnz_cols, indptr, indices, idtype, sorted)  # type: ignore


def dense_fixed_axis(name: str, length: PrimExpr, idtype: str) -> Axis:
    """Dense-fixed axis creator.

    Parameters
    ----------
    name : str
        The name of the axis

    length : PrimExpr
        The length of the axis

    idtype : str
        The index data type
    """
    return Axis(name, None, length, length, length, None, None, idtype, True)


def dense_variable_axis(
    name: str, parent: Axis, length: PrimExpr, nnz: PrimExpr, indptr: Var, idtype: str
) -> Axis:
    """Dense-variable axis creator.

    Parameters
    ----------
    name : str
        The name of the axis

    parent : Axis
        The parent axis

    length : PrimExpr
        The length of the axis

    nnz : PrimExpr
        The accumulated number of nonzero elements from root axis to current axis.

    indptr : Var
        The indptr buffer var of the axis

    idtype : str
        The index data type
    """
    return Axis(name, parent, length, nnz, None, indptr, None, idtype, True)


def sparse_fixed_axis(
    name: str,
    parent: Axis,
    length: PrimExpr,
    nnz_cols: PrimExpr,
    indices: Var,
    idtype: str,
    sorted: bool = True,
) -> Axis:
    """Sparse-fixed axis creator.

    Parameters
    ----------
    name : str
        The name of the axis

    parent : Axis
        The parent axis

    length : PrimExpr
        The length of the axis

    indices : Var
        The indices buffer var of the axis

    nnz_cols : PrimExpr
        The fixed number of non-zero elements along the axis

    idtype : str
        The index data type

    sorted : bool
        The indices are sorted or not.
    """
    return Axis(
        name, parent, length, parent.nnz * nnz_cols, nnz_cols, None, indices, idtype, sorted
    )


def sparse_variable_axis(
    name: str,
    parent: Axis,
    length: PrimExpr,
    nnz: PrimExpr,
    indptr: Var,
    indices: Var,
    idtype: str,
    sorted: bool = True,
) -> Axis:
    """Sparse-variable axis creator.

    Parameters
    ----------
    name : str
        The name of the axis

    parent : Axis
        The parent axis

    length : PrimExpr
        The length of the axis

    nnz : PrimExpr
        The number of non zero elements in the axis.

    indptr : Var
        The indptr buffer var of the axis

    indices : Var
        The indices buffer var of the axis

    idtype : str
        The index data type

    sorted : bool
        The indices are sorted or not.
    """
    return Axis(name, parent, length, nnz, None, indptr, indices, idtype, sorted)


@tvm._ffi.register_object("tir.sparse.FusedAxis")
class FusedAxis(Axis):
    """FusedAxis node

    Parameters
    ----------
    group : List[Axis]
        The axes group to be fused.
    index : int
        The index of current axis in the fused axes group.
    """

    group: List[Axis]
    index: int

    def __init__(self, group, index):
        self.__init_handle_by_constructor__(_ffi_api.FusedAxis, group, index)  # type: ignore


@tvm._ffi.register_object("tir.sparse.FlattenedAxis")
class FlattenedAxis(Axis):
    """FlattenedAxis node

    Parameters
    ----------
    name : str
        The name of the axis.

    axes : List[Axis]
        The axes to flatten.

    nnz : PrimExpr
        The number of nonzeros of the returned axis.

    offsets : List[Var]
        The list of new offset arrays.

    idtype : str
        The index data type.
    """

    name: str
    axes: List[Axis]
    nnz: PrimExpr
    offsets: List[Var]
    idtype: str

    def __init__(self, name, axes, nnz, offsets, idtype):
        self.__init_handle_by_constructor__(
            _ffi_api.FlattenedAxis, name, axes, nnz, offsets, idtype
        )


@tvm._ffi.register_object("tir.sparse.AttachedAxis")
class AttachedAxis(Axis):
    """AttachedAxis node

    Parameters
    ----------
    base : Axis
        The base axis.
    new_parent : Axis
        The new parent axis to attach.
    """

    base: Axis
    new_parent: Axis

    def __init__(self, base, new_parent):
        self.__init_handle_by_constructor__(_ffi_api.AttachedAxis, base, new_parent)  # type: ignore


@tvm._ffi.register_object("tir.sparse.SparseBuffer")
class SparseBuffer(Buffer):
    """SparseBuffer node

    Parameters
    ----------
    data : Var
        The pointer to underlying data.
    axes : List[Axis]
        The axes of the sparse buffer
    dtype : str
        The data type of this sparse buffer.
    name : str
        The name of the sparse buffer
    extra_storage : Optional[PrimExpr]
        Required extra storage (e.g. for indptr)
    default_value : Optional[PrimExpr]
        The default value about missing value of the the sparse buffer
    span : Span
    """

    data: Var
    axes: List[Axis]
    dtype: str
    name: str
    extra_storage: Optional[PrimExpr]
    default_value: Optional[PrimExpr]
    span: Span

    def __init__(self, data, axes, dtype, name, extra_storage, default_value, span):
        self.__init_handle_by_constructor__(_ffi_api.SparseBuffer, data, axes, dtype, name, extra_storage, default_value, span)  # type: ignore


@tvm._ffi.register_object("tir.sparse.SpIterVar")
class SpIterVar(Object):
    """IterVar in SparseTIR

    Parameters
    ----------
    var : Var
        The var of the SpIterVar

    is_reduction : bool
        Whether the SpIterVar is a reduction iterator

    axis : Axis
        The axis over which the SpIterVar iterates
    """

    var: Var
    is_reduction: bool
    axis: Axis

    def __init__(self, var, is_reduction, axis):
        self.__init_handle_by_constructor__(
            _ffi_api.SpIterVar, var, is_reduction, axis  # type: ignore
        )
