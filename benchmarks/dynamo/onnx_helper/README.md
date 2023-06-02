This folder contains python scripts to extend the dynamo benchmarking framework for benchmarking ONNX export. Bash scripts to run the benchmark and generate markdown reports are also provided.

# Usage

## Setup

It is recommended to create a fresh python environment, clone and build PyTorch from source.

Then install the benchmark dependencies:

```bash
./0_build_bench.sh
```

## Run Benchmark

```bash
./1_bench.sh
```

## Generate Report

After benchmark completes. Run the following script to generate a markdown report and archive the benchmark results.

```bash
./2_generate_report_and_archive.sh
```

## Bench single model

```bash
./bench_single_model.sh BertForMaskedLM
```