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

"""Format module for sparse tensor algebra."""
from typing import Callable, Dict, List, Union
import tvm._ffi
import tvm.tir

from tvm.runtime import Object
from tvm.tir import IndexMap, _ffi_api
from tvm import IRModule
from tvm.tir.transform import SparseFormatDecompose


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

    idx_map_func : Union[Callable, IndexMap]
        A function describing the index mapping from the old format to indices in new format.

    inv_idx_map_func : Union[Callable, IndexMap]
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
        idx_map: Union[Callable, IndexMap],
        inv_idx_map: Union[Callable, IndexMap],
    ) -> None:
        if isinstance(idx_map, Callable):
            idx_map = IndexMap.from_func(idx_map)
        if isinstance(inv_idx_map, Callable):
            inv_idx_map = IndexMap.from_func(inv_idx_map)
        self.__init_handle_by_constructor__(
            _ffi_api.FormatRewriteRule,
            name,
            new_format_desc,
            buffers_to_rewrite,
            axes_before_rewrite,
            axes_after_rewrite,
            axis_map,
            idx_map,
            inv_idx_map,
        )  # type: ignore


def column_part_hyb(num_rows, num_cols, indptr_nd, indices_nd, num_col_parts, buckets):
    """Partition input CSR matrix by columns and collect rows into buckets according to non zero elements per row.

    Parameters
    ----------
    num_rows : int
        Number of rows in the CSR matrix.
    num_cols : int
        Number of columns in the CSR matrix.
    indptr : NDArray
        The indptr array of CSR matrix.
    indices : NDArray
        The indices array of CSR matrix.
    num_col_parts : int
        Number of column partitions.
    buckets : List
        The bucket sizes array.

    Returns
    -------
    Tuple[List[List[NDArray]]]
        The pair of (row_indices, col_indices, mask).
        row_indices is stored as a list of lists with shape (num_col_parts, len(buckets)), where the innermost element is an NDArray.
        col_indices and mask are stored in the same way.
    """
    return _ffi_api.ColumnPartHyb(
        num_rows, num_cols, indptr_nd, indices_nd, num_col_parts, buckets  # type: ignore
    )


def condense(indptr_nd, indices_nd, t, g):
    """Condense sparse matrix in CSR format to (t x 1) tiles, and group g tiles together.


    Parameters
    ----------
    indptr : NDArray
        The indptr array of CSR format.
    indices : NDArray
        The indices array of CSR format.
    t : int
        The tile size.
    g : int
        The group size.

    Returns
    -------
    Tuple[NDArray]
        The pair of (group_indptr, tile_indices, mask).
    """
    return _ffi_api.ConDense(indptr_nd, indices_nd, t, g)  # type: ignore


def csf_to_ell3d(
    csf_indptr_0, csf_indices_0, csf_indptr_1, csf_indices_1, nnz_rows_bkt, nnz_cols_bkt
):
    """Convert CSF format to composable ELL format in 3-dimensional setting (HeteroGraphs).

    Parameters
    ----------
    csf_indptr_0 : NDArray
        Level 0 indptr array in CSF format.
    csf_indices_0 : NDArray
        Level 0 indices array in CSF format.
    csf_indptr_1 : NDArray
        Level 1 indptr array in CSF format.
    csf_indices_1 : NDArray
        Level 1 indices array in CSF format.
    num_rows_bkt : List[int]
        Number of non-zero rows bucket.
    nnz_cols_bkt : List[int]
        Number of non-zero columns bucket.

    Returns
    -------
    Tuple[List[NDArray]]
        (indptr, row_indices, col_indices, mask)
        Each one is a list of NDArray, with length #rels.
    """
    return _ffi_api.CSFToELL3D(
        csf_indptr_0, csf_indices_0, csf_indptr_1, csf_indices_1, nnz_rows_bkt, nnz_cols_bkt
    )


def format_decompose(
    mod: IRModule,
    composable_formats: List["FormatRewriteRule"],
    include_format_rewrite_blks: bool = True,
):
    """Rewrite the sparse format of sparse buffers in the TIR scripts.

    Parameters
    ----------
    mod : IRModule
        The IRModule to lower.
    composable_formats : List[FormatRewriteRule]
        Composable formats is a list of rewrite rules.
    include_format_rewrite_blks : bool
        Whether to include format rewrite blocks in the output.
    """
    if not isinstance(mod, IRModule):
        raise TypeError("Expected IRModule, but got {}".format(type(mod)))
    return SparseFormatDecompose(composable_formats, include_format_rewrite_blks)(mod)
