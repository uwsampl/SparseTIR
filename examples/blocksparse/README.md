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
