# Inductor, inference
TORCH_LOGS="-dynamo,-aot,-inductor" python benchmarks/dynamo/huggingface.py --performance --timing --explain --backend inductor --device cuda >data/hf_inductor_eval.log 2>&1 
TORCH_LOGS="-dynamo,-aot,-inductor" python benchmarks/dynamo/timm_models.py --performance --timing --explain --backend inductor --device cuda >data/timm_inductor_eval.log 2>&1 
TORCH_LOGS="-dynamo,-aot,-inductor" python benchmarks/dynamo/torchbench.py --performance --timing --explain --backend inductor --device cuda >data/tb_inductor_eval.log 2>&1 

# NVFuser, inference
TORCH_LOGS="-dynamo,-aot,-inductor" python benchmarks/dynamo/huggingface.py --performance --timing --explain --backend nvprims_nvfuser --device cuda >data/hf_nvfuser_eval.log 2>&1 
TORCH_LOGS="-dynamo,-aot,-inductor" python benchmarks/dynamo/timm_models.py --performance --timing --explain --backend nvprims_nvfuser --device cuda >data/timm_nvfuser_eval.log 2>&1 
TORCH_LOGS="-dynamo,-aot,-inductor" python benchmarks/dynamo/torchbench.py --performance --timing --explain --backend nvprims_nvfuser --device cuda >data/tb_nvfuser_eval.log 2>&1 

# Inductor, training
TORCH_LOGS="-dynamo,-aot,-inductor" python benchmarks/dynamo/huggingface.py --performance --timing --explain --training --backend inductor --device cuda >data/hf_inductor_train.log 2>&1 
TORCH_LOGS="-dynamo,-aot,-inductor" python benchmarks/dynamo/timm_models.py --performance --timing --explain --training --backend inductor --device cuda >data/timm_inductor_train.log 2>&1 
TORCH_LOGS="-dynamo,-aot,-inductor" python benchmarks/dynamo/torchbench.py --performance --timing --explain --training --backend inductor --device cuda >data/tb_inductor_train.log 2>&1 

# NVFuser, training
TORCH_LOGS="-dynamo,-aot,-inductor" python benchmarks/dynamo/huggingface.py --performance --timing --explain --training --backend nvprims_nvfuser --device cuda >data/hf_nvfuser_train.log 2>&1 
TORCH_LOGS="-dynamo,-aot,-inductor" python benchmarks/dynamo/timm_models.py --performance --timing --explain --training --backend nvprims_nvfuser --device cuda >data/timm_nvfuser_train.log 2>&1 
TORCH_LOGS="-dynamo,-aot,-inductor" python benchmarks/dynamo/torchbench.py --performance --timing --explain --training --backend nvprims_nvfuser --device cuda >data/tb_nvfuser_train.log 2>&1 