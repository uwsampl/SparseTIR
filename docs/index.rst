..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

.. SparseTIR documentation master file, created by
   sphinx-quickstart on Wed Jul 20 16:15:56 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SparseTIR's documentation!
=====================================

SparseTIR is a tensor-level compiler for sparse/irregular operators in Deep Learning. The design goal of SparseTIR is to provide a general programming abstraction that can cover both sparse and irregular (e.g. Ragged Tensors) workloads in Deep Learning including Graph Neural Networks, Sparse Transformers, Sparse Convolutions, Network Pruning, etc. while generating high-performance code on heterogeneous hardware.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   install.rst
   tutorials/blitz.rst

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/spmm.rst
   tutorials/sddmm.rst
   tutorials/spmm-tc.rst
   tutorials/blocksparse.rst
   tutorials/rgcn.rst
   tutorials/sparseconv.rst

.. toctree::
   :maxdepth: 1
   :caption: System Overview

   motivation.rst
   overview.rst

.. toctree::
   :maxdepth: 1
   :caption: Misc

   faq.rst

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/python/tvm.sparse