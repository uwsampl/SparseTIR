# Sparse Dense Matrix Multiplication

This folder contain SparseTIR SpMM implementation w/ and w/o composable formats, we also show how to formulate TC-GNN's con-densing format and schedule in SparseTIR.

```
spmm
|   bench_spmm_naive.py       # SpMM in SparseTIR w/o composable formats.
|   bench_spmm.py             # SpMM in SparseTIR w/ composable formats.
|   bench_spmm_tc.py          # SpMM in SparseTIR using Tensor Cores (equivalent to TC-GNN paper: https://arxiv.org/pdf/2112.02052.pdf)
```