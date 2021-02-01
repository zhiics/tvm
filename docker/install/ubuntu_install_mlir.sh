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

set -e
set -u
set -o pipefail

WORKDIR=/mlir
mkdir ${WORKDIR}
cd ${WORKDIR}
LLVMDIR=${WORKDIR}/llvm-project
LLVMBUILD=${WORKDIR}/llvm-build
MLIRDIR=${WORKDIR}/mlir-hlo

apt-get update && apt-get install -y ninja-build lld-9
ln -s /usr/bin/lld-9 /usr/bin/lld
ln -s /usr/bin/ld.lld-9 /usr/bin/ld.lld

# Install llvm-project from source
git clone https://github.com/llvm/llvm-project.git
mkdir ${LLVMBUILD}

# Install mlir
cd ${WORKDIR}
git clone https://github.com/tensorflow/mlir-hlo.git
cd ${LLVMDIR}
git checkout $(cat ${MLIRDIR}/build_tools/llvm_version.txt)
cd ${MLIRDIR}
chmod +x build_tools/build_mlir.sh

build_tools/build_mlir.sh ${LLVMDIR} ${LLVMBUILD}

mkdir build && cd build

cmake .. -GNinja \
  -DLLVM_ENABLE_LLD=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=On \
  -DMLIR_DIR=${LLVMBUILD}/lib/cmake/mlir
