KERNEL_DIR = "./data_tb"

import subprocess, os
from os import listdir
from os.path import isfile, join, isdir

seen_kernels = set()

for model in sorted(listdir(KERNEL_DIR)):
    model_path = join(KERNEL_DIR, model)
    if not isdir(model_path):
        continue

    if model < "amp_training_timm_efficientnet":
        for kernel in sorted(listdir(model_path)):
            kernel_path = join(model_path, kernel)
            if not isdir(kernel_path):
                continue
            for py in listdir(kernel_path):
                if py.endswith(".py"):
                    seen_kernels.add(py[:-3])
                    print("Add seen " + py[:-3])
        continue

    for kernel in sorted(listdir(model_path)):
        kernel_path = join(model_path, kernel)
        if not isdir(kernel_path):
            continue

        # remove best config file
        for py in listdir(kernel_path):
            py_path = join(kernel_path, py)
            if py.endswith((".best_config", ".all_config", ".log")):
                cmd = "rm -rf " + py_path
                print(cmd)
                os.system(cmd)

        # run kernel
        for py in listdir(kernel_path):
            py_path = join(kernel_path, py)
            if not py.endswith(".py"):
                continue
            
            # skip graph python file
            with open(py_path, "r") as file:
                content = file.read()
                if "Original ATen:" in content:
                    print("Skip " + py_path + " GRAPH")
                    continue

            if py[:-3] in seen_kernels:
                print("Skip " + py_path + " <<<<<< " + py[:-3] + " seen before")
                continue

            cache_dir = kernel_path
            log_path = join(kernel_path, py[:-3] + ".log")
            cmd = """CUDA_VISIBLE_DEVICES=1 TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=1 TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1 TORCHINDUCTOR_CACHE_DIR=[[CACHE_DIR]] TORCH_LOGS="+inductor" TORCHINDUCTOR_BENCHMARK_KERNEL=1 TORCHINDUCTOR_COORDINATE_DESCENT_RADIUS=2 TORCHINDUCTOR_COORDINATE_DESCENT_CHECK_ALL_DIRECTIONS=1 python3 [[PY_PATH]] --device-iddd 1 > [[LOG_PATH]] 2>&1"""
            cmd = (
                cmd.replace("[[CACHE_DIR]]", cache_dir)
                .replace("[[PY_PATH]]", py_path)
                .replace("[[LOG_PATH]]", log_path)
            )
            print(cmd)
            try:
                subprocess.run(cmd, timeout=90, shell=True)
            except subprocess.TimeoutExpired as exc:
                print(exc)

            seen_kernels.add(py[:-3])
