.. _apidata

tvm.sparse
==========

.. currentmodule:: tvm.sparse
.. automodule:: tvm.sparse

Program transformations
---------------------------------------

.. autosummary::
  :toctree: ../../generated

  lower_sparse_iter
  lower_sparse_buffer
  format_decompose
  specialize_buffer

Composable Format Description
---------------------------------------

.. autosummary::
  :toctree: ../../generated

  FormatRewriteRule

Format Conversion Routines
---------------------------------------

.. autosummary::
  :toctree: ../../generated

  condense
  column_part_hyb
  csf_to_ell3d