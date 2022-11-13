# Block Sparse Operators in SparseTIR


```
.
blocksparse
|   README.md
|   bsr_spmm.py             # SpMM for block sparse format
|   bsr_sddmm.py            # SDDMM for block sparse format
|   bsr_sparse_softmax.py   # Sparse Softmax for block sparse format
```

We haven't enabled software pipelining and async copy, nor any optimizations to avoid bank conflicts yet, and the performance could be further improved.
