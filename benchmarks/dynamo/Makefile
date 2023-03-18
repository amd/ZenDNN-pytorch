# Makefile for the dashboard setup
TRITON_VERSION ?= $(shell cat ../../.github/ci_commit_pins/triton.txt)
PIP ?= python -m pip

clone-deps:
	(cd ../../.. \
		&& (test -e torchvision || git clone --recursive https://github.com/pytorch/vision torchvision) \
		&& (test -e torchdata || git clone --recursive https://github.com/pytorch/data.git torchdata) \
		&& (test -e torchtext || git clone --recursive https://github.com/pytorch/text torchtext) \
		&& (test -e torchaudio || git clone --recursive https://github.com/pytorch/audio torchaudio) \
		&& (test -e detectron2 || git clone --recursive https://github.com/facebookresearch/detectron2) \
		&& (test -e torchbenchmark || git clone --recursive https://github.com/pytorch/benchmark torchbenchmark) \
		&& (test -e triton || (git clone --recursive https://github.com/openai/triton.git && git checkout $(TRITON_VERSION) && git submodule update --init --recursive)) \
	)

pull-deps: clone-deps
	echo $(TRITON_VERSION)
	(cd ../../../torchvision    && git reset --hard HEAD && git pull && git submodule update --init --recursive)
	(cd ../../../torchdata      && git reset --hard HEAD && git pull && git submodule update --init --recursive)
	(cd ../../../torchtext      && git reset --hard HEAD && git pull && git submodule update --init --recursive)
	(cd ../../../torchaudio     && git reset --hard HEAD && git pull && git submodule update --init --recursive)
	(cd ../../../detectron2     && git reset --hard HEAD && git pull && git submodule update --init --recursive)
	(cd ../../../torchbenchmark && git reset --hard HEAD && git pull && git submodule update --init --recursive)
	(cd ../../../triton         && git reset --hard HEAD && git fetch && git checkout $(TRITON_VERSION) && git submodule update --init --recursive)

build-deps:
	(cd ../../../torchvision && python setup.py clean && python setup.py develop)
	(cd ../../../torchdata && python setup.py install)
	(cd ../../../torchtext   && python setup.py clean && python setup.py develop)
	(cd ../../../torchaudio   && python setup.py clean && python setup.py develop)
	(cd ../../../detectron2  && python setup.py clean && python setup.py develop)
	(cd ../../../torchbenchmark && python install.py --continue_on_fail)
	(cd ../../../triton/python && python setup.py clean && python setup.py develop)
