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
