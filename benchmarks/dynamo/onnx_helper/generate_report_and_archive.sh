#!/bin/bash
# Run this script under pytorch/benchmarks/dynamo/onnx_helper after 'bench.sh' completes.
# This script generates ONNX benchmark error summary report in markdown.
# It expects to find benchmark logs under pytorch/.logs/onnx_bench/benchmark_logs.
# It will generate markdown reports under that folder.
# When it's done, it will archive the benchmark logs and rename the folder
# to pytorch/.logs/onnx_bench/benchmark_logs_<timestamp>.

set -e

pushd "../../../"

log_folder=".logs/onnx_bench/benchmark_logs"

python benchmarks/dynamo/onnx_helper/reporter.py \
    --suites=torchbench \
    --suites=huggingface \
    --suites=timm_models \
    --compilers dynamo-onnx \
    --compilers onnx \
    --output-dir=./"$log_folder"

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
new_log_folder="$log_folder"_"$timestamp"

# TODO: stitch all gh logs together and publish to github

mv "$log_folder" "$new_log_folder"

popd