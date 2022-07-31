Install SparseTIR
=================

Currently we only support build SparseTIR from source code.

Build from Source
-----------------

  .. code:: bash

    git clone --recursive git@github.com:uwsampl/sparsetir.git
    mkdir build
    cd build
    cmake .. -DUSE_CUDA=ON -DUSE_LLVM=ON
    make -j$(nproc)