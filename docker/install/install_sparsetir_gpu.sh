#!/bin/bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Clear ccache
ccache -C

# Compile C++ source code.
echo set\(USE_LLVM \"llvm-config-15 --ignore-libllvm --link-static\"\) >> config.cmake
echo set\(HIDE_PRIVATE_SYMBOLS ON\) >> config.cmake
echo set\(USE_CUDA ON\) >> config.cmake
echo set\(USE_CUBLAS ON\) >> config.cmake
echo set\(USE_CUDNN ON\) >> config.cmake
mkdir -p build
cd build
cmake ..
make -j$(nproc)
cd ..

# Install Python binding.
cd python/
pip3 install -e .
