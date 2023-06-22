DB_DIRS = ["./data_hf", "./data_tb", "./data_timm"]

import os
import pickle
import torch
import subprocess
import logging
from os import listdir
from os.path import isfile, join, isdir
from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.dependencies import StarDep, WeakDep

logger = logging.getLogger()


kernel_counter = 0
seen_kernels = set()
op_counter = 0
op_dict = dict()


class KernelCategory:
    POINTWISE = 0
    REDUCTION = 1
    PERSISTENT_REDUCTION = 2


def get_kernel_category(model: str, kernel: str, src: str) -> KernelCategory:
    if "@pointwise" in src:
        return KernelCategory.POINTWISE
    if "@reduction" in src:
        return KernelCategory.REDUCTION
    if "@persistent_reduction" in src:
        return KernelCategory.PERSISTENT_REDUCTION


def get_number_of_loops(model: str, kernel: str, src: str) -> int:
    return src.count("for roffset in range(0, rnumel, RBLOCK):")
    

def parse_list_of_numbers(s: str) -> list:
    # num1, num2, num3, ...
    nums = s.strip().split(",")
    nums = [num.strip() for num in nums]
    return [int(num) for num in nums]


def get_size_hints(model: str, kernel: str, src: str) -> list:
    startpos = src.find("size_hints=[")
    assert startpos != -1
    endpos = src.find("]", startpos)
    return parse_list_of_numbers(src[startpos + len("size_hints=[") : endpos])
    
    
for DB_DIR in DB_DIRS:
    for model in sorted(listdir(DB_DIR)):
        model_path = join(DB_DIR, model)
        if not isdir(model_path):
            continue

        for kernel in sorted(listdir(model_path)):
            kernel_path = join(model_path, kernel)
            if not isdir(kernel_path):
                continue

            for py in listdir(kernel_path):
                py_path = join(kernel_path, py)
                if not py.endswith(".py"):
                    continue

                with open(py_path, "r") as file:
                    src = file.read()
                    if "Original ATen:" in src:
                        continue

                kernel_name = py[:-3]
                log_path = join(kernel_path, kernel_name + ".log")
                pkl_path = join(kernel_path, py + ".pkl")
                all_config_path = join(kernel_path, kernel_name + ".all_config")

                # Get the kernel category
                # Some kernels are just eliminated by the compiler
                kernel_category = get_kernel_category(model, kernel_name, src)
                if kernel_category is None:
                    continue
                
                # Sanity check
                if os.path.exists(log_path) and os.path.exists(pkl_path) and os.path.exists(all_config_path):
                    seen_kernels.add(kernel_name)
                elif not os.path.exists(log_path) or not os.path.exists(pkl_path) or not os.path.exists(all_config_path):
                    if not kernel_name in seen_kernels:
                        logger.warning(f"Missing {model}, {kernel_name}")
                    continue
                else:
                    logger.warning(f"Incomplete {model}, {kernel_name}")
                    continue

                # Get the number of loops
                if kernel_category is KernelCategory.REDUCTION:
                    num_of_loops = get_number_of_loops(model, kernel_name, src)
                else:
                    num_of_loops = 0

                kernel_counter = kernel_counter + 1
                
                # Map the ops to numbers
                (reads, writes, total_bytes), nodes, node_read_writes, src_code = tuple(pickle.load(open(pkl_path, "rb")))
                op_counts = node_read_writes.op_counts
                op_bag = dict()
                for op in op_counts.keys():
                    if op in op_dict:
                        op_bag[op_dict[op]] = op_counts[op]
                    else:
                        op_dict[op] = op_counter
                        op_bag[op_counter] = op_counts[op]
                        op_counter = op_counter + 1
                print(model, kernel_name, kernel_category, num_of_loops)
                print(op_bag)
                
                # Get the size hints from src code
                size_hints = get_size_hints(model, kernel_name, src_code)
                
                # Get the stride and shape vec from flattened read_writes
                # sort the reads/writes according to the names of buf
                reads = sorted(reads, key=lambda x: x[0])
                writes = sorted(writes, key=lambda x: x[0])
                print(size_hints)
                sizevar_allocator = SizeVarAllocator()
                for dep, bytes in zip(reads + writes, total_bytes):
                    if isinstance(dep, (StarDep, WeakDep)):
                        print("StarDep or WeakDep")
                        continue
                    print(dep.index, dep.var_names, bytes)
                    strides = sizevar_allocator.stride_hints(dep.index, dep.var_names)
                    print(strides)
                    for stride in strides:
                        assert isinstance(stride, int)

print(op_dict)
print(kernel_counter)
