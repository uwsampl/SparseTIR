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

Currently we only support build SparseTIR from source code.

Build from Source
-----------------

  The first step is to compile source code written in C++.

  .. code:: bash

    git clone --recursive git@github.com:uwsampl/sparsetir.git
    mkdir build
    cd build
    cmake .. -DUSE_CUDA=ON -DUSE_LLVM=ON
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

