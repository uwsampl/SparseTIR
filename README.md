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
[Documentation](https://sampl.cs.washington.edu/sparsetir/) |
[Paper](https://arxiv.org/abs/2207.04606)

SparseTIR is a tensor-level compiler for sparse/irregular operators in Deep Learning. The design goal of SparseTIR is to provide a general programming abstraction that can cover both sparse and irregular (e.g. Ragged Tensors) sparse workloads in Deep Learning including Graph Neural Networks, Sparse Transformers, Sparse Convolutions, Network Pruning, etc. while generating high-performance code on heterogeneous hardware.

The key innovation of SparseTIR is *composability*:
- **Format Composability**: Decompose the computation on a single format to computation on hybrid formats.
- **Transformation Composability**: SparseTIR adopts multi-stage IR design on top of TVM's TensorIR, user can compose program transformations at different stages to bring together sparsity-aware optimizations such as format searching and low-level optimizations such as vectorization and tensorization.

Check out the [Documentation](https://sampl.cs.washington.edu/sparsetir/) site for installation instructions, tutorials, examples, and more. The [Blitz Introduction to SparseTIR](https://sampl.cs.washington.edu/sparsetir/tutorials/blitz.html) is a great place to get you familiar with SparseTIR's format annotations and compilation flow. You can check the tutorials on optimizing [SpMM](https://sampl.cs.washington.edu/sparsetir/tutorials/spmm.html) and [RGMS](https://sampl.cs.washington.edu/sparsetir/tutorials/rgcn.html) to understand how to use composable formats and composable transformations to optimize sparse operators.

This repo is still experimental and documentations are under construction, the API will not be stable and we will frequently refactor the codebase. However, we believe making it public will benefit researchers and engineers in this field. Feel free to create an issue if you run into any problems or have any suggestions on SparseTIR. We plan to contribute SparseTIR to Apache TVM project as an optional module, and we will keep maintaining this repo until the core functionalities have been upstreamed.

Acknowledgement
---------------
SparseTIR was built upon [TensorIR](https://arxiv.org/pdf/2207.04296.pdf) in [Apache TVM](https://tvm.apache.org/), we thank all the generous help from the TVM community. We also learned a lot from the [TACO](https://github.com/tensor-compiler/taco) project: the format annotations and the concept of position space and coordinate space come from TACO. We thank the TACO team for their foundation work in sparse compilers.

Related Work
------------
You can also check out the following cool projects if you are interested in our work:
- [MLIR-Sparse](https://mlir.llvm.org/docs/Dialects/SparseTensorOps/) MLIR's sparse tensor dialect.
- [SparTA](https://github.com/microsoft/SparTA) SparTA is an abstraction for model sparsity in Deep Learning.

