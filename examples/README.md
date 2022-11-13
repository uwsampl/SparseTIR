# SparseTIR examples

The folder contains example SparseTIR implementations of typical sparse operators in Deep Learning.
- [SpMM](spmm/): Sparse-Dense Matrix Multiplication
  - We introduce how to use composable formats/transformations to optimize SpMM in SparseTIR, we also demonstrate how to formulate [TC-GNN](https://arxiv.org/pdf/2112.02052.pdf) in SparseTIR.
- [SDDMM](sddmm/): Sampled Dense Dense Matrix Multiplication
  - We demonstrate how to use composable transformations to formulate [PRedS](https://ieeexplore.ieee.org/document/9643711) in SparseTIR, and perform some parameter search for optimization.
- [Block Sparse](blocksparse/): Sparse operators on block sparse format
  - Sparse operators on block sparse formats.
- [RGMS](rgms/): Relational Gather-Matmul-Scatter.
  - Notable examples of RGMS are Relational Graph Convolutional Networks ([RGCN](https://arxiv.org/pdf/1703.06103.pdf)) and [Sparse Convolution](https://arxiv.org/pdf/1904.08755.pdf) for point cloud processing.
  - We show how to fuse Gather, Matrix Multiplication and Scatter in a single kernel and uses SparseTIR's composable formats/transformations to optimize it.


More examples are coming, including [FusedMM](https://arxiv.org/pdf/2011.06391.pdf)+[FlashAttention](https://arxiv.org/pdf/2205.14135.pdf) for Sparse Matrix.

We welcome contributions from community, please create a [pull request](https://github.com/uwsampl/sparsetir/pulls) if you find a better schedule for any of existing examples or have a SparseTIR implementation of new sparse operators.
