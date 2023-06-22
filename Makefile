#*******************************************************************************
# Modifications Copyright (c) 2023 Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
#*******************************************************************************

# This makefile does nothing but delegating the actual building to cmake.
PYTHON = python3
PIP = pip3

all:
	@mkdir -p build && cd build && cmake .. $(shell $(PYTHON) ./scripts/get_python_cmake_flags.py) && $(MAKE)

local:
	@./scripts/build_local.sh

android:
	@./scripts/build_android.sh

ios:
	@./scripts/build_ios.sh

clean: # This will remove ALL build folders.
	@rm -r build*/

linecount:
	@cloc --read-lang-def=caffe.cloc caffe2 || \
		echo "Cloc is not available on the machine. You can install cloc with " && \
		echo "    sudo apt-get install cloc"

setup_lint:
	@if [ "$$(uname)" = "Darwin" ]; then \
		if [ -z "$$(which brew)" ]; then \
			echo "'brew' is required to install ShellCheck, get it here: https://brew.sh "; \
			exit 1; \
		fi; \
		brew install shellcheck; \
	else \
		$(PYTHON) tools/actions_local_runner.py --file .github/workflows/lint.yml \
		--job 'shellcheck' --step 'Install ShellCheck' --no-quiet; \
	fi
	pip install typing-extensions==3.10
	pip install -r requirements-flake8.txt
	flake8 --version
	pip3 install cmakelint==1.4.1
	cmakelint --version
	python3 -mpip install -r requirements.txt
	python3 -mpip install numpy==1.20
	python3 -mpip install expecttest==0.1.3 mypy==0.812
	# Needed to check tools/render_junit.py
	python3 -mpip install junitparser==2.1.1 rich==10.9.0
	pip install Jinja2==3.0.1

	$(PYTHON) -mpip install jinja2
	$(PYTHON) -mpip install -r tools/linter/clang_tidy/requirements.txt
	$(PYTHON) -m tools.linter.install.clang_tidy

quick_checks:
# TODO: This is broken when 'git config submodule.recurse' is 'true' since the
# lints will descend into third_party submodules
	@$(PYTHON) tools/actions_local_runner.py \
		--file .github/workflows/lint.yml \
		--job 'quick-checks' \
		--step 'Ensure no trailing spaces' \
		--step 'Ensure no tabs' \
		--step 'Ensure no non-breaking spaces' \
		--step 'Ensure canonical include' \
		--step 'Ensure no versionless Python shebangs' \
		--step 'Ensure no unqualified noqa' \
		--step 'Ensure GitHub PyPi dependencies are pinned' \
		--step 'Ensure no unqualified type ignore' \
		--step 'Ensure no direct cub include' \
		--step 'Ensure correct trailing newlines' \
		--step 'Ensure no raw cuda api calls'

flake8:
	@$(PYTHON) tools/actions_local_runner.py \
		$(CHANGED_ONLY) \
		--job 'flake8-py3'

mypy:
	@$(PYTHON) tools/actions_local_runner.py \
		$(CHANGED_ONLY) \
		--job 'mypy'

cmakelint:
	@$(PYTHON) tools/actions_local_runner.py \
		--file .github/workflows/lint.yml \
		--job 'cmakelint' \
		--step 'Run cmakelint'

clang-tidy:
	@$(PYTHON) tools/actions_local_runner.py \
		$(CHANGED_ONLY) \
		--job 'clang-tidy'

toc:
	@$(PYTHON) tools/actions_local_runner.py \
		--file .github/workflows/lint.yml \
		--job 'toc' \
		--step "Regenerate ToCs and check that they didn't change"

lint:
	lintrunner

quicklint:
	lintrunner
