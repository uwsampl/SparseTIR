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

# Sparse Dense Matrix Multiplication

This folder contain SparseTIR SpMM implementation w/ and w/o composable formats, we also show how to formulate TC-GNN's con-densing format and schedule in SparseTIR.

```
spmm
|   bench_spmm_naive.py       # SpMM in SparseTIR w/o composable formats.
|   bench_spmm.py             # SpMM in SparseTIR w/ composable formats.
|   bench_spmm_tc.py          # SpMM in SparseTIR using Tensor Cores (equivalent to TC-GNN paper: https://arxiv.org/pdf/2112.02052.pdf)
```