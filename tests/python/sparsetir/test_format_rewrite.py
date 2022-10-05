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

import tvm
import dgl
import numpy as np
from tvm.sparse import FormatRewriteRule, column_part_hyb, condense
from sparse_tir_scripts import csrmm
from sparse_tir_format_rewrite_scripts import (
    bsr,
    bsr_rewrite_with_preprocess,
    ell,
    ell_rewrite_with_preprocess,
    padding,
    padding_rewrite_with_preprocess,
)


def csr2bsr_inv_index_map(block_size):
    def func(io, jo, ii, ji):
        return io * block_size + ii, jo * block_size + ji

    return func


def csr2bsr_index_map(block_size):
    def func(i, j):
        return i // block_size, j // block_size, i % block_size, j % block_size

    return func


def csr2ell_inv_index_map(o, i, j):
    return i, j


def csr2ell_index_map(i, j):
    return 0, i, j


def test_csrmm_bsr_rewrite():
    block_size_symbol = bsr.params[-1]
    rewrites = []
    for block_size in [4, 16, 32]:
        rewrites.append(
            FormatRewriteRule(
                str(block_size),
                bsr.specialize({block_size_symbol: block_size}),
                ["A"],
                ["I", "J"],
                ["IO", "JO", "II", "JI"],
                {"I": ["IO", "II"], "J": ["JO", "JI"]},
                csr2bsr_index_map(block_size),
                csr2bsr_inv_index_map(block_size),
            )
        )
    mod = tvm.IRModule.from_expr(csrmm)
    mod = tvm.tir.transform.SparseFormatRewrite(rewrites)(mod)
    print(mod["main"].script())
    tvm.ir.assert_structural_equal(
        mod["main"], bsr_rewrite_with_preprocess, True)


def test_csrmm_ell_rewrite():
    nnz_cols_symbol = ell.params[-1]
    rewrites = []
    for nnz_cols in [4, 8, 16, 32, 64, 128, 512]:
        rewrites.append(
            FormatRewriteRule(
                str(nnz_cols),
                ell.specialize({nnz_cols_symbol: nnz_cols}),
                ["A"],
                ["I", "J"],
                ["O", "I", "J"],
                {"I": ["O", "I"], "J": ["J"]},
                csr2ell_index_map,
                csr2ell_inv_index_map,
            )
        )
    mod = tvm.IRModule.from_expr(csrmm)
    mod = tvm.tir.transform.SparseFormatRewrite(rewrites)(mod)
    tvm.ir.assert_structural_equal(
        mod["main"], ell_rewrite_with_preprocess, True)


def csrpadding_inv_index_map(i, jo, ji):
    return i, ji


def csrpadding_index_map(i, j):
    return i, 0, j


def test_csrmm_padding_rewrite():
    pad_size_symbol = padding.params[-1]
    pad_size = 32
    rewrites = [
        FormatRewriteRule(
            str(pad_size),
            padding.specialize({pad_size_symbol: pad_size}),
            ["A"],
            ["I", "J"],
            ["I", "JO", "JI"],
            {"I": ["I"], "J": ["JO", "JI"]},
            csrpadding_index_map,
            csrpadding_inv_index_map,
        )
    ]
    mod = tvm.IRModule.from_expr(csrmm)
    mod = tvm.tir.transform.SparseFormatRewrite(rewrites)(mod)
    tvm.ir.assert_structural_equal(
        mod["main"], padding_rewrite_with_preprocess, True)


def scipy_column_part_hyb(g, column_part, bucket_sizes):
    mat = g.adj(transpose=True, scipy_fmt="csr")
    buckets = bucket_sizes * column_part
    m = mat.shape[0]
    n = mat.shape[1]
    nnz = mat.nnz
    per_column_part_size = (n + column_part - 1) // column_part
    sub_mats = [
        mat[:, i * per_column_part_size: (i + 1) * per_column_part_size]
        for i in range(column_part)
    ]

    num_buckets = len(buckets)
    ell_n = []

    for partition in range(column_part):
        sub_mat = sub_mats[partition]
        in_degrees = sub_mat.indptr[1:] - sub_mat.indptr[:-1]
        for i, bucket_size in enumerate(bucket_sizes[:-1]):
            last_bucket_size = 0 if i == 0 else bucket_sizes[i - 1]
            ell_n.append(int(((in_degrees > last_bucket_size)
                         & (in_degrees <= bucket_size)).sum()))
        sub_indegrees = in_degrees[in_degrees > bucket_sizes[-2]]
        ell_n.append(
            int(((sub_indegrees + bucket_sizes[-1] - 1) // bucket_sizes[-1]).sum()))

        ell_rows = []
        ell_indices = []

    for partition in range(column_part):
        sub_mat = sub_mats[partition]
        in_degrees = sub_mat.indptr[1:] - sub_mat.indptr[:-1]

        for i, bucket_size in enumerate(bucket_sizes[:-1]):
            last_bucket_size = 0 if i == 0 else bucket_sizes[i - 1]
            ell_rows.append(
                ((in_degrees > last_bucket_size) &
                 (in_degrees <= bucket_size)).nonzero()[0]
            )
        ell_rows.append((in_degrees > bucket_sizes[-2]).nonzero()[0])

        for i, bucket_size in enumerate(bucket_sizes[:-1]):
            indices = np.zeros(
                (ell_n[partition * len(bucket_sizes) + i],
                 bucket_size), dtype=np.int32
            )
            for j, row_id in enumerate(ell_rows[partition * len(bucket_sizes) + i]):
                row = sub_mat[row_id]
                indices[j, : row.nnz] = row.indices + \
                    partition * per_column_part_size
            ell_indices.append(indices)

        # split rows for the last bucket
        indices = np.zeros(
            (ell_n[(partition + 1) * len(bucket_sizes) - 1],
             bucket_sizes[-1]), dtype=np.int32
        )
        new_rows = np.zeros(
            (ell_n[(partition + 1) * len(bucket_sizes) - 1],), dtype=np.int32)
        bucket_size = bucket_sizes[-1]
        i = 0
        for row_id in ell_rows[-1]:
            row = sub_mat[row_id]
            for start_offset in range(0, row.nnz, bucket_size):
                if start_offset + bucket_size >= row.nnz:
                    # last bucket
                    indices[i, : row.nnz - start_offset] = (
                        row.indices[start_offset:] +
                        partition * per_column_part_size
                    )
                else:
                    indices[i] = (
                        row.indices[start_offset: start_offset + bucket_size]
                        + partition * per_column_part_size
                    )
                new_rows[i] = row_id
                i += 1

        ell_indices.append(indices)
        ell_rows[-1] = new_rows

    return ell_rows, ell_indices


def test_column_part_hyb():
    g = dgl.rand_graph(1000, 10000).int()
    column_parts = 4
    buckets = [1, 2, 4]
    indptr, indices, _ = g.adj_sparse("csc")
    indptr_nd = tvm.nd.array(indptr.numpy(), device=tvm.cpu())
    indices_nd = tvm.nd.array(indices.numpy(), device=tvm.cpu())
    # built-in c++ funcion
    row_indices, col_indices, mask = column_part_hyb(
        g.num_dst_nodes(), g.num_src_nodes(), indptr_nd, indices_nd, column_parts, buckets
    )
    # compute indices with scipy
    row_indices_scipy, col_indices_scipy = scipy_column_part_hyb(
        g, column_parts, buckets)

    for part_id in range(column_parts):
        for bucket_id, _ in enumerate(buckets):
            assert np.array_equal(
                row_indices[part_id][bucket_id].numpy(),
                row_indices_scipy[part_id * len(buckets) + bucket_id],
            )
            assert np.array_equal(
                col_indices[part_id][bucket_id].numpy(),
                col_indices_scipy[part_id * len(buckets) + bucket_id],
            )


def condense_py(indptr, indices, block_size):
    m = len(indptr) - 1
    ret_indptr = [0]
    ret_indices = []
    for block_id in range((m + block_size - 1) // block_size):
        start_offset = indptr[block_id * block_size]
        end_offset = indptr[-1] if (block_id + 1) * \
            block_size > m else indptr[(block_id + 1) * block_size]
        tile_indices = indices[start_offset: end_offset]
        unique_col_indices = np.unique(tile_indices)
        ret_indptr.append(ret_indptr[-1] + len(unique_col_indices))
        ret_indices.append(unique_col_indices)
    return np.array(ret_indptr), np.concatenate(ret_indices)


def test_condense():
    g = dgl.rand_graph(1000, 10000).int()
    t = 4 
    indptr, indices, _ = g.adj_sparse("csc")
    indptr = indptr.numpy()
    indices = indices.numpy()
    indptr_nd = tvm.nd.array(indptr, device=tvm.cpu())
    indices_nd = tvm.nd.array(indices, device=tvm.cpu())
    # built-in c++ function
    indptr_ret, indices_ret, mask = condense(indptr_nd, indices_nd, t, 1)
    # Python version of function
    indptr_py, indices_py = condense_py(indptr, indices, t)
    assert np.array_equal(
        indptr_ret.numpy().flatten(),
        indptr_py,
    )
    assert np.array_equal(
        indices_ret.numpy().flatten(),
        indices_py,
    )


if __name__ == "__main__":
    test_csrmm_bsr_rewrite()
    test_csrmm_ell_rewrite()
    test_csrmm_padding_rewrite()
    test_column_part_hyb()
    test_condense()
