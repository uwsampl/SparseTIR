# Relational Gather-Matmul-Scatter

This folder contains examples (RGCN and Sparse Convolution) of optimizing Relational Gather-Matmul-Scatter(RGMS) with SparseTIR.

```
rgms
├── rgcn
|     bench_rgcn_baseline.py            # rgcn implementations in DGL/PyG
|     bench_rgcn_non_composable.py      # fuse gather-matmul-scatter in SparseTIR, without using composable format/tensor cores.
|     bench_rgcn_composable.py          # fuse gather-matmul-scatter in SparseTIR, using composable format, no tensor cores.
|     bench_rgcn_tensorcore.py          # fuse gather-matmul-scatter in SparseTIR, using composable format and tensor cores.
├── sparse_conv
|     sparse_conv.py                    # apply SparseTIR's rgms operator to Sparse Convolution.
```

We haven't enabled software pipelining and async copy, nor any optimizations to avoid bank conflicts yet, and the performance could be further improved.
