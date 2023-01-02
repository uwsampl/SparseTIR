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

Install SparseTIR
=================

Currently we only support build SparseTIR from source code, you'll need to build the shared library (which is written in C++) first and then install Python bindings.
It's worth noting that SparseTIR is a fork of Apache TVM project, and you don't need to install Apache TVM to use Sparse TIR.

Pre-requisites
--------------

  We recommend user to install following packages before compiling SparseTIR shared library:

  - A recent C++ compiler supporting C++ 14.
  - CMake 3.18 or higher
  - LLVM 10 or higher
  - CUDA Toolkit 11 or higher
  - Python 3.9 or higher


Build the Shared Library
------------------------

  The first step is to compile source code written in C++.

  .. code:: bash

    git clone --recursive git@github.com:uwsampl/sparsetir.git
    echo set\(USE_LLVM \"llvm-config --ignore-libllvm --link-static\"\) >> config.cmake
    echo set\(HIDE_PRIVATE_SYMBOLS ON\) >> config.cmake
    echo set\(USE_CUDA ON\) >> config.cmake
    echo set\(USE_CUBLAS ON\) >> config.cmake
    echo set\(USE_CUDNN ON\) >> config.cmake
    mkdir -p build
    cd build 
    cmake ..
    make -j$(nproc)

Install Python Binding
----------------------

  If compilation is successful, the next step is to install SparseTIR binding for Python, you can either install Python package via:

  .. code:: bash

    cd python
    python3 setup.py install
  

  or set environment variable `${PYTHONPATH}`:

  .. code:: bash

    export SPARSETIR_PATH=$(pwd)
    export PYTHONPATH=${SPARSETIR_PATH}/python:${PYTHONPATH}


