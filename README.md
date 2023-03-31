<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->


SparseTIR: Sparse Tensor Compiler for Deep Learning
==============================================
[Documentation](https://sampl.cs.washington.edu/SparseTIR/) |
[Paper]([https://arxiv.org/abs/2207.04606](https://dl.acm.org/doi/10.1145/3582016.3582047)

[![Build Status](https://github.com/uwsampl/sparsetir/actions/workflows/build.yml/badge.svg)](https://github.com/uwsampl/sparsetir/actions/workflows/build.yml)

SparseTIR is a tensor-level compiler for sparse/irregular operators in Deep Learning. The design goal of SparseTIR is to provide a general programming abstraction that can cover both sparse and irregular (e.g. Ragged Tensors) workloads in Deep Learning including Graph Neural Networks, Sparse Transformers, Sparse Convolutions, Network Pruning, etc. while generating high-performance code on heterogeneous hardware.

The key innovation of SparseTIR is *composability*:
- **Format Composability**: Decompose the computation on a single format to computation on hybrid formats.
- **Transformation Composability**: SparseTIR adopts multi-stage IR design on top of TVM's TensorIR, user can compose program transformations at different stages to bring together sparsity-aware optimizations such as format searching and low-level optimizations such as vectorization and tensorization.

Check out the [Documentation](https://sampl.cs.washington.edu/SparseTIR/) site for installation instructions, tutorials, examples, and more. The [Blitz Introduction to SparseTIR](https://sampl.cs.washington.edu/SparseTIR/tutorials/blitz.html) is a great place to get you familiar with SparseTIR's format annotations and compilation flow. You can check the tutorials on optimizing [SpMM](https://sampl.cs.washington.edu/SparseTIR/tutorials/spmm.html) and [RGMS](https://sampl.cs.washington.edu/SparseTIR/tutorials/rgcn.html) to understand how to use composable formats and composable transformations to optimize sparse operators.

This repo is still experimental and documentations are under construction, the API will not be stable and we will frequently refactor the codebase. However, we believe making it public will benefit researchers and engineers in this field. Feel free to create an issue if you run into any problems or have any suggestions on SparseTIR. We are upstreaming SparseTIR to Apache TVM mainline, check [our RFC](https://discuss.tvm.apache.org/t/rfc-sparsetir-as-a-new-dialect-in-tvm/14645) for the plan.

Contributors
------------
The codebase is mainly developed and maintained by [Ruihang Lai](https://github.com/MasterJH5574) and [Zihao Ye](https://github.com/yzh119/), we also thank the following wonderful contributors:

- [Yinuo Liu](https://github.com/qelk123)
- [Sidharth Lakshmanan](https://github.com/sidlak-c137)

Acknowledgement
---------------
SparseTIR was built upon [TensorIR](https://arxiv.org/pdf/2207.04296.pdf) in [Apache TVM](https://tvm.apache.org/), we thank all the generous help from the TVM community. We also learned a lot from the [TACO](https://github.com/tensor-compiler/taco) project: the format annotations and the concept of position space and coordinate space come from TACO. We thank the TACO team for their foundation work in sparse compilers.

Related Work
------------
You can also check out the following cool projects if you are interested in our work:
- [MLIR-Sparse](https://mlir.llvm.org/docs/Dialects/SparseTensorOps/) MLIR's sparse tensor dialect.
- [SparTA](https://github.com/microsoft/SparTA) SparTA is an abstraction for model sparsity in Deep Learning.

Cite SparseTIR
--------------
If you find the design of SparseTIR useful or use SparseTIR in your work, you can consider citing SparseTIR paper with the bibtex below:
```bibtex
@inproceedings{10.1145/3582016.3582047,
  author = {Ye, Zihao and Lai, Ruihang and Shao, Junru and Chen, Tianqi and Ceze, Luis},
  title = {SparseTIR: Composable Abstractions for Sparse Compilation in Deep Learning},
  year = {2023},
  isbn = {9781450399180},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3582016.3582047},
  doi = {10.1145/3582016.3582047},
  abstract = {Sparse tensors are rapidly becoming critical components of modern deep learning workloads. However, developing high-performance sparse operators can be difficult and tedious, and existing vendor libraries cannot satisfy the escalating demands from new operators. Sparse tensor compilers simplify the development of operators, but efficient sparse compilation for deep learning remains challenging because a single sparse format cannot maximize hardware efficiency, and single-shot compilers cannot keep up with latest hardware and system advances. In this paper, we observe that the key to addressing both these challenges is to leverage composable formats and composable transformations. We propose SparseTIR, a sparse tensor compilation abstraction that offers composable formats and composable transformations for deep learning workloads. SparseTIR constructs a search space over these composable components for performance tuning. With these improvements, SparseTIR obtains consistent performance speedups vs vendor libraries on GPUs for single operators: 1.20-2.34x for GNN operators, 1.05-2.98x for sparse attention operators, and 0.56-7.45x for sparse convolution operators. SparseTIR also accelerates end-to-end GNNs by 1.08-1.52x for GraphSAGE training, and 4.20-40.18x for RGCN inference.},
booktitle = {Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3},
  pages = {660–678},
  numpages = {19},
  keywords = {Scheduling, Vectorization, Kernel Fusion, Code Generation and Optimizations, Tensor Cores, Tensor Compilers, Sparse Computation},
  location = {Vancouver, BC, Canada},
  series = {ASPLOS 2023}
}
```
