Sparse Convolution
==================

The convolution operator can be formulated as:

.. math::
    :label: eq:sp-conv
  
    \mathbf{Y}_{n, fout, y, x} = \sum_{i=0}^{K-1}\sum_{j=0}^{K-1}\sum_{fin=0}^{D_{in}} \mathbf{W}_{fout, fin, y, x} \mathbf{X}_{n, fin, y+j, x+i}

where :math:`n` iterates over each instance in a batch, :math:`x` and :math:`y` iterates over columns and rows.

The sparsity of convolution could come from model weights (weight sparsity) or from the features (activation sparsity), some paper studies the configuration of dual sparsity where both weights and activations are sparse, however, acceleration of dual sparsity on GPU architecture requires specific acceleration unit support (e.g. `Dual-Side Sparse Tensor Core <https://arxiv.org/pdf/2105.09564.pdf>`_), and we only focus on either weight/activation sparsity in this tutorial.

Weight Sparsity
---------------

Sparse convolution with weight sparsity was proposed in paper `Faster CNNs with Direct Sparse Convolutions and Guided Prunning <https://arxiv.org/pdf/1608.01409.pdf>`_, where they store the :math:`W` in formula `eq:sp-conv`_ as a sparse matrix.

Activation Sparsity
-------------------

`torchsparse <https://github.com/mit-han-lab/torchsparse>`_ and `spconv <https://github.com/traveller59/spconv>`_ are two libraries optimizing convolution of activation sparsity. 