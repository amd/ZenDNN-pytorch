#!/bin/bash
# Run this script under pytorch/benchmarks/dynamo/onnx_helper to start benchmarking onnx w/ torchbench.
# This script generates ONNX benchmark report logs under pytorch/.logs/onnx_bench.
# It is expected to further run "generate_report_and_archive.sh" after this script completes.

# NOTE: use 'nohup' and add '&' to the end to prevent script stopping due to terminal timeout.

set -e

pushd "../../../"

log_folder=".logs/onnx_bench"
terminal_log="bench_terminal.log"

echo "Running benchmarking onnx w/ torchbench in background..."
echo "Benchmark logs will be saved under pytorch/$log_folder"
echo "Terminal logs are redirected to pytorch/$terminal_log"
echo "Please run 'generate_report_and_archive.sh' after this script completes."
echo "This script is run with 'nohup' and '&' to prevent script stopping due to terminal timeout."
echo "You can monitor the progress by inspecting pytorch/$terminal_log or running 'tail -f pytorch/$terminal_log'"

# TODO: it depends but we can make 'nohup' optional for easier debugging.
# NOTE: --quick is handy to run on small subset of ~3 models for quick sanity check.

PATH=/usr/local/cuda/bin/:$PATH nohup time python benchmarks/dynamo/runner.py \
    --suites=torchbench \
    --suites=huggingface \
    --suites=timm_models \
    --inference \
    --batch_size 1 \
    --compilers dynamo-onnx \
    --compilers onnx \
    --dashboard-image-uploader None \
    --dashboard-archive-path "$log_folder"/cron_logs \
    --dashboard-gh-cli-path None \
    --output-dir "$log_folder"/benchmark_logs > "$terminal_log" 2>&1 &

popd