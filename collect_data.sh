#!/bin/bash

set -x #echo

# Inductor, inference
# python benchmarks/dynamo/huggingface.py --performance --timing --explain --backend inductor --device cuda >data/hf_inductor_eval.log 2>&1 
# python benchmarks/dynamo/timm_models.py --performance --timing --explain --backend inductor --device cuda >data/timm_inductor_eval.log 2>&1 
# python benchmarks/dynamo/torchbench.py --performance --timing --explain --backend inductor --device cuda >data/tb_inductor_eval.log 2>&1 

# NVFuser, inference
# python benchmarks/dynamo/huggingface.py --performance --timing --explain --backend nvprims_nvfuser --device cuda >data/hf_nvfuser_eval.log 2>&1 
# python benchmarks/dynamo/timm_models.py --performance --timing --explain --backend nvprims_nvfuser --device cuda >data/timm_nvfuser_eval.log 2>&1 
# python benchmarks/dynamo/torchbench.py --performance --timing --explain --backend nvprims_nvfuser --device cuda >data/tb_nvfuser_eval.log 2>&1 

# Inductor, training
# python benchmarks/dynamo/huggingface.py --performance --timing --explain --training --backend inductor --device cuda >data/hf_inductor_train.log 2>&1 
# python benchmarks/dynamo/timm_models.py --performance --timing --explain --training --backend inductor --device cuda >data/timm_inductor_train.log 2>&1 
# python benchmarks/dynamo/torchbench.py --performance --timing --explain --training --backend inductor --device cuda >data/tb_inductor_train.log 2>&1 

# NVFuser, training
# python benchmarks/dynamo/huggingface.py --performance --timing --explain --training --backend nvprims_nvfuser --device cuda >data/hf_nvfuser_train.log 2>&1 
# python benchmarks/dynamo/timm_models.py --performance --timing --explain --training --backend nvprims_nvfuser --device cuda >data/timm_nvfuser_train.log 2>&1 
# python benchmarks/dynamo/torchbench.py --performance --timing --explain --training --backend nvprims_nvfuser --device cuda >data/tb_nvfuser_train.log 2>&1 