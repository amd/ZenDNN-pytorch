#!/bin/bash

set -e

# Move to pytorch/
pushd "../../../"

# Sync and build pytorch
git fetch origin
git rebase origin/viable/strict
git submodule update --init --recursive
USE_CUDA=1 python setup.py develop

popd

# Move to pytorch/benchmarks/dynamo
pushd "../"

# Clone and sync repos
make pull-deps

# build repos
(cd ../../../torchvision && python setup.py clean && python setup.py develop)
(cd ../../../torchdata && python setup.py install)
(cd ../../../torchtext   && python setup.py clean && python setup.py develop)
(cd ../../../torchaudio   && python setup.py clean && python setup.py develop)
(cd ../../../detectron2  && python setup.py clean && python setup.py develop)
(cd ../../../torchbenchmark && python install.py --continue_on_fail)
(cd ../../../triton/python && python setup.py clean && python setup.py develop)

popd

# Install onnx dependencies
# TODO: Ideally the same as CI environment. Script is repeated here.

# TODO: use official onnx package once it's released
# for now, use the commit from 1.13.1-protobuf4.21 branch
pip install "onnx@git+https://github.com/onnx/onnx@389b6bcb05b9479d149d29b2461fbffe8472ed14"
pip install ort-nightly-gpu==1.15.0.dev20230502003 --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/
pip install "onnxscript@git+https://github.com/microsoft/onnxscript@99a665546d374af872cf136137b5d8334389b78d"
