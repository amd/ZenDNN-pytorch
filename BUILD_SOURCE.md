**`Documentation`** |
------------------- |
To build PyTorch with ZenDNN follow below steps.

## Build From Source
### Setup for Linux
Create and activate a conda environment and install the following dependencies
```
$ conda install ninja pyyaml cmake cffi typing_extensions future six requests dataclasses astunparse setuptools numpy
$ conda install cpuonly -c pytorch
```


### Download the AMD ZenDNN PyTorch source code
Location of AMD ZenDNN PyTorch: [AMD ZenDNN PyTorch](https://github.com/amd/ZenDNN-pytorch).

Checkout AMD ZenDNN PyTorch
```
$ git clone https://github.com/amd/ZenDNN-pytorch.git
$ cd ZenDNN-pytorch
```

The repo defaults to the main development branch which doesnt has ZenDNN support. You need to check out a release branch to build, e.g. `release/1.12_zendnn_rel` or `release/1.13_zendnn_rel` etc.
```
$ git checkout branch_name  # release/1.12_zendnn_rel, release/1.13_zendnn_rel, etc.
```


### Set environment variables
Set environment variables for optimum performance. Some of the environment variables are for housekeeping purposes and can be ignored.
```
$ source scripts/zendnn_PT_env_setup.sh
```


### Download the ZenDNN and AOCL-BLIS into third_party
Location of ZenDNN: [ZenDNN](https://github.com/amd/ZenDNN).

Location of AOCL-BLIS: [AOCL-BLIS](https://github.com/amd/blis).

```
$ cd third_party/
$ git clone https://github.com/amd/ZenDNN
$ cd ZenDNN
$ git checkout v4.1
$ cd ../
$ git clone https://github.com/amd/blis.git
$ cd blis
$ git checkout 4.1
$ cd ../../
```


### Update submodules in pytorch
```
$ git submodule sync
$ git submodule update --init --recursive
```


### Build and install the pip package
```
$ export PYTORCH_BUILD_NUMBER=1
$ export PYTORCH_BUILD_VERSION=1.13.1
$ python setup.py clean
$ USE_ROCM=0 USE_ZENDNN=1 USE_ZENDNN_QUANT=1 python setup.py bdist_wheel
$ USE_ROCM=0 USE_ZENDNN=1 USE_ZENDNN_QUANT=1 python setup.py install
```


### Quick verification of Build. You should see ZenDNN and BLIS versions in following prints
### Change directory to parent folder
```
$ cd ../
$ python -c 'import torch; print(*torch.__config__.show().split("\n"), sep="\n")'
```