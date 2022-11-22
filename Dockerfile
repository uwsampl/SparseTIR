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

# CI docker GPU env
# tag: v0.60
FROM nvidia/cuda:11.6.0-cudnn8-devel-ubuntu20.04 as base

ENV DEBIAN_FRONTEND=noninteractive

# Per https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212772
# we need to add a new GPG key before running apt update.
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

# Base scripts
RUN apt-get clean
RUN apt-get update --fix-missing
RUN apt-get -y install cmake

COPY docker/install/ubuntu_install_core.sh /install/ubuntu_install_core.sh
RUN bash /install/ubuntu_install_core.sh

COPY docker/install/ubuntu2004_install_python.sh /install/ubuntu2004_install_python.sh
RUN bash /install/ubuntu2004_install_python.sh

# Globally disable pip cache
RUN pip config set global.no-cache-dir false

COPY docker/install/ubuntu2004_install_llvm.sh /install/ubuntu2004_install_llvm.sh
RUN bash /install/ubuntu2004_install_llvm.sh

COPY docker/install/ubuntu_install_python_package.sh /install/ubuntu_install_python_package.sh
RUN bash /install/ubuntu_install_python_package.sh

COPY docker/install/ubuntu_install_sphinx.sh /install/ubuntu_install_sphinx.sh
RUN bash /install/ubuntu_install_sphinx.sh

COPY docker/install/ubuntu_install_dgl.sh /install/ubuntu_install_dgl.sh
RUN bash /install/ubuntu_install_dgl.sh

COPY docker/install/ubuntu_install_torch.sh /install/ubuntu_install_torch.sh
RUN bash /install/ubuntu_install_torch.sh

COPY docker/install/ubuntu_install_rat.sh /install/ubuntu_install_rat.sh
RUN bash /install/ubuntu_install_rat.sh

# Environment variables
ENV PATH=/usr/local/nvidia/bin:${PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV CPLUS_INCLUDE_PATH=/usr/local/cuda/include:${CPLUS_INCLUDE_PATH}
ENV C_INCLUDE_PATH=/usr/local/cuda/include:${C_INCLUDE_PATH}
ENV LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat:${LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat:${LD_LIBRARY_PATH}

# Ensure the local libcuda have higher priority than the /usr/local/cuda/compact
# since the compact libcuda does not work on non-Tesla gpus
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH}
ENV PATH=/node_modules/.bin:${PATH}

# Install SparseTIR GPU
WORKDIR /root/sparsetir
ADD .git/ .git/
ADD 3rdparty 3rdparty/
ADD cmake cmake/
ADD configs configs/
ADD include include/
ADD python python/
ADD src src/
ADD tests tests/
COPY CMakeLists.txt CMakeLists.txt
COPY docker/install/install_sparsetir_gpu.sh /install/install_sparsetir_gpu.sh
RUN bash /install/install_sparsetir_gpu.sh
ENV PYTHONPATH=python/:${PYTHONPATH}

# Add other folders
ADD docs docs/
ADD conftest.py conftest.py
ADD gallery gallery/
ADD golang golang/
ADD jenkins jenkins/
ADD jvm jvm/
ADD tvm-docs tvm-docs/
ADD vta vta/
ADD conda conda/
ADD apps apps/
ADD examples examples/
ADD configs configs/
ADD version.py version.py
ADD web web/

# Install dependencies required by lint
RUN apt install -y clang-format
RUN pip3 install flake8==5.0.4 pylint==2.15.6 cpplint==1.6.1 black==22.8.0
RUN apt install -y default-jre
