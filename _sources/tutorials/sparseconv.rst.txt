Sparse Convolution
==================

The computation of sparse convolution can be formulated as:

.. math::
  
    \mathbf{Y}_{n, fout, y, x} = \sum_{i=0}^{K-1}\sum_{j=0}^{K-1}\sum_{fin=0}^{D_{in}} \mathbf{W}_{fout, fin, y, x} \mathbf{X}_{n, fin, y+j, x+i}

Where :math:`n` iterates over each instance in a batch, :math:`\mathbf{W}` is a sparse matrix.