#!/bin/bash
# Run this script under pytorch/benchmarks/dynamo/onnx_helper to start benchmarking onnx w/ torchbench.
# This script generates ONNX benchmark report logs under pytorch/.logs/onnx_bench.

set -e

# Check if the number of arguments is less than 1
if [ $# -lt 1 ]; then
    echo "Error: No argument provided."
    echo "Usage: ./bench_single_model.sh <model_name>"
    exit 1
fi

model="$1"

pushd "../../../"

log_folder=".logs/onnx_bench"

echo "Running benchmarking onnx w/ torchbench in background..."
echo "Benchmark logs will be saved under pytorch/$log_folder"

PATH=/usr/local/cuda/bin/:$PATH time python benchmarks/dynamo/runner.py \
    --suites=torchbench \
    --suites=huggingface \
    --suites=timm_models \
    --inference \
    --batch_size 1 \
    --compilers dynamo-onnx \
    --extra-args "--filter $model" \
    --dashboard-image-uploader None \
    --dashboard-archive-path "$log_folder"/cron_logs \
    --dashboard-gh-cli-path None \
    --output-dir "$log_folder"/benchmark_logs

popd
