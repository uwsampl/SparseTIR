.. SparseTIR documentation master file, created by
   sphinx-quickstart on Wed Jul 20 16:15:56 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SparseTIR's documentation!
=====================================

SparseTIR is a tensor-level compiler for sparse/irregular operators in Deep Learning. The design goal of SparseTIR is to provide a general programming abstraction that can cover both sparse and irregular (e.g. Ragged Tensors) sparse workloads in Deep Learning including Graph Neural Networks, Sparse Transformers, Sparse Convolutions, Network Pruning, etc. while generating high-performance code on heterogeneous hardware.

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

   overview.rst

.. toctree::
   :maxdepth: 1
   :caption: Migration Plan

   migration.rst

API doc
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
