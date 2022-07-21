Install SparseTIR
=================

Build from Source
-----------------

  .. code:: bash

    git clone --recursive git@github.com:uwsampl/sparsetir.git
    mkdir build
    cd build
    cp ../cmake.config .
    cmake ..
    make -j$(nproc)