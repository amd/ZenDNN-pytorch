import copy
import os

LOG_DIR = "./data-logs-hf/"

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

model_names = []

with open("/scratch/bohanhou/fresh/pytorch/benchmarks/dynamo/huggingface_models_list.txt", "r") as f:
    for line in f.readlines():
        line = line.split(",")[0]
        model_names.append(line)

print(model_names)

template = """TORCHINDUCTOR_CACHE_DIR=/scratch/bohanhou/fresh/data_hf/[[DTYPE]]_[[MODE]]_[[MODEL_NAME]] TORCH_LOGS="+inductor" TORCHINDUCTOR_BENCHMARK_KERNEL=1 python3 benchmarks/dynamo/huggingface.py --[[DTYPE]] --performance --[[MODE]] --inductor -d cuda --device-index 0 --filter [[MODEL_NAME]] """ +  " >" + LOG_DIR + "[[DTYPE]]_[[MODE]]_[[MODEL_NAME]].kernels.log 2>&1"

for DTYPE in ["amp"]:
    for MODE in ["training"]:
        for model_name in model_names:
            cmd = copy.deepcopy(template)
            cmd = cmd.replace("[[MODEL_NAME]]", model_name).replace("[[MODE]]", MODE).replace("[[DTYPE]]", DTYPE)
            print(cmd)
            
            os.system(cmd)
