# SparseTIR examples

The folder contains example SparseTIR implementations of typical sparse operators in Deep Learning.
- [SpMM](spmm/): Sparse-Dense Matrix Multiplication
  - We introduce how to use composable formats/transformations to optimize SpMM in SparseTIR, we also demonstrate how to formulate TC-GNN in SparseTIR.
- [SDDMM](sddmm/): Sampled Dense Dense Matrix Multiplication
  - We demonstrate how to use composable transformations to formulate PRedS in SparseTIR, and perform some parameter search for optimization.
- [Block Sparse](blocksparse/): Sparse operators on block sparse format
  - This folder includes sparse operators on block sparse formats.
- [RGMS](rgms/): Relational Gather-Matmul-Scatter.
  - Notable examples of RGMS are Relational Graph Convolutional Networks (RGCN) and 3D Sparse Convolution for point cloud processing.
  - We show how to fuse Gather, Matrix Multiplication and Scatter in a single kernel and uses SparseTIR's composable formats/transformations to optimize it.


More examples are coming, including FusedMM+FlashAttention for Sparse Matrix.