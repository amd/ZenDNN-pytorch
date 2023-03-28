# Jason's comment
# Similar to what the --coverage option of the benchmark script reports.
# 1) How many models work (dynamo only)
# 2) Number of graphs per model
# 3) Number of ops captured/not captured/percentage

# We could compare with torchscript as baseline for #1


#!/bin/bash
set -x
# Setup the output directory
rm -rf ../benchmark_logs
mkdir -p ../benchmark_logs

# Coverage numbers
python ../benchmarks/dynamo/torchbench.py --performance --float32 -dcuda --output=benchmark_logs/tb_dynamo.csv   --no-skip --dashboard -x pyhpc_equation_of_state -x detectron2_maskrcnn_r_101_c4 -x detectron2_fasterrcnn_r_50_fpn -x opacus_cifar10 -x detectron2_fasterrcnn_r_50_c4 -x detectron2_fasterrcnn_r_101_dc5 -x pyhpc_turbulent_kinetic_energy -x fambench_xlmr -x timm_efficientdet -x detectron2_fasterrcnn_r_101_fpn -x detectron2_maskrcnn_r_50_fpn -x maml -x pyhpc_isoneutral_mixing -x detectron2_maskrcnn_r_101_fpn -x detectron2_maskrcnn -x detectron2_fasterrcnn_r_101_c4 -x detectron2_fasterrcnn_r_50_dc5 -x DALLE2_pytorch -x moco -x detectron2_fcos_r_50_fpn -x torchrec_dlrm
python ../benchmarks/dynamo/huggingface.py --performance --float32 -dcuda --output=benchmark_logs/hf_dynamo.csv   --no-skip --dashboard -x BlenderbotForConditionalGeneration -x GPTJForCausalLM -x GPTNeoForSequenceClassification -x GPTNeoForCausalLM -x GPTJForQuestionAnswering -x Reformer
python ../benchmarks/dynamo/timm_models.py --performance --float32 -dcuda --output=benchmark_logs/timm_dynamo.csv   --no-skip --dashboard -x eca_halonext26ts -x levit_128



# Torchbench vs Dynamo
python ../benchmarks/dynamo/torchbench.py --performance --float32 -dcuda --output=benchmark_logs/tb_ts.csv   --no-skip --dashboard -x pyhpc_equation_of_state -x detectron2_maskrcnn_r_101_c4 -x detectron2_fasterrcnn_r_50_fpn -x opacus_cifar10 -x detectron2_fasterrcnn_r_50_c4 -x detectron2_fasterrcnn_r_101_dc5 -x pyhpc_turbulent_kinetic_energy -x fambench_xlmr -x timm_efficientdet -x detectron2_fasterrcnn_r_101_fpn -x detectron2_maskrcnn_r_50_fpn -x maml -x pyhpc_isoneutral_mixing -x detectron2_maskrcnn_r_101_fpn -x detectron2_maskrcnn -x detectron2_fasterrcnn_r_101_c4 -x detectron2_fasterrcnn_r_50_dc5  -x DALLE2_pytorch -x moco -x detectron2_fcos_r_50_fpn -x torchrec_dlrm --run-ts-nodynamo
python ../benchmarks/dynamo/huggingface.py --performance --float32 -dcuda --output=benchmark_logs/hf_ts.csv   --no-skip --dashboard -x BlenderbotForConditionalGeneration -x GPTJForCausalLM -x GPTNeoForSequenceClassification -x GPTNeoForCausalLM -x GPTJForQuestionAnswering -x Reformer5 --run-ts-nodynamo
python ../benchmarks/dynamo/timm_models.py --performance --float32 -dcuda --output=benchmark_logs/timm_ts.csv   --no-skip --dashboard -x eca_halonext26ts -x levit_1285 --run-ts-nodynamo

# Overhead numbers

python ../benchmarks/dynamo/torchbench.py --performance --float32 -dcuda --output=benchmark_logs/tb_overhead.csv   --no-skip --dashboard -x pyhpc_equation_of_state -x detectron2_maskrcnn_r_101_c4 -x detectron2_fasterrcnn_r_50_fpn -x opacus_cifar10 -x detectron2_fasterrcnn_r_50_c4 -x detectron2_fasterrcnn_r_101_dc5 -x pyhpc_turbulent_kinetic_energy -x fambench_xlmr -x timm_efficientdet -x detectron2_fasterrcnn_r_101_fpn -x detectron2_maskrcnn_r_50_fpn -x maml -x pyhpc_isoneutral_mixing -x detectron2_maskrcnn_r_101_fpn -x detectron2_maskrcnn -x detectron2_fasterrcnn_r_101_c4 -x detectron2_fasterrcnn_r_50_dc5 -x DALLE2_pytorch -x moco -x detectron2_fcos_r_50_fpn -x torchrec_dlrm --overhead
python ../benchmarks/dynamo/huggingface.py --performance --float32 -dcuda --output=benchmark_logs/hf_overhead.csv   --no-skip --dashboard -x BlenderbotForConditionalGeneration -x GPTJForCausalLM -x GPTNeoForSequenceClassification -x GPTNeoForCausalLM -x GPTJForQuestionAnswering -x Reformer --overhead
python ../benchmarks/dynamo/timm_models.py --performance --float32 -dcuda --output=benchmark_logs/timm_overhead.csv   --no-skip --dashboard -x eca_halonext26ts -x levit_128 --overhead
