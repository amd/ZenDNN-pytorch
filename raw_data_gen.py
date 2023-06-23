DB_DIRS = ["./data_hf", "./data_tb", "./data_timm"]

import os
import copy
import pickle
import torch
import json
import pickle
import subprocess
import logging
from os import listdir
from os.path import isfile, join, isdir
from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.dependencies import StarDep, WeakDep

logger = logging.getLogger()


kernel_counter = 0
config_counter = 0
seen_kernels = set()
op_counter = 0

# op_dict needs to be deterministic
op_dict = {
    "load": 0,
    "to_dtype": 1,
    "add": 2,
    "reduction": 3,
    "constant": 4,
    "div": 5,
    "store": 6,
    "sub": 7,
    "square": 8,
    "rsqrt": 9,
    "mul": 10,
    "tanh": 11,
    "ne": 12,
    "where": 13,
    "indirect_indexing": 14,
    "log": 15,
    "neg": 16,
    "exp": 17,
    "maximum": 18,
    "minimum": 19,
    "index_expr": 20,
    "ge": 21,
    "masked": 22,
    "lt": 23,
    "and_": 24,
    "erf": 25,
    "eq": 26,
    "le": 27,
    "gt": 28,
    "relu": 29,
    "sqrt": 30,
    "logical_not": 31,
    "load_seed": 32,
    "rand": 33,
    "abs": 34,
    "reciprocal": 35,
    "ceil": 36,
    "sigmoid": 37,
    "sin": 38,
    "cos": 39,
    "logical_and": 40,
    "bitwise_and": 41,
    "randn": 42,
    "floor": 43,
    "remainder": 44,
    "isinf": 45,
    "logical_or": 46,
    "expm1": 47,
    "libdevice_sqrt": 48,
    "libdevice_log": 49,
    "truediv": 50,
    "sign": 51,
    "randint64": 52,
    "bitwise_or": 53,
    "pow": 54,
}


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


raw_data = list()

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
                if (
                    os.path.exists(log_path)
                    and os.path.exists(pkl_path)
                    and os.path.exists(all_config_path)
                ):
                    seen_kernels.add(kernel_name)
                elif (
                    not os.path.exists(log_path)
                    or not os.path.exists(pkl_path)
                    or not os.path.exists(all_config_path)
                ):
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

                cur_kernel_dict = dict()
                cur_kernel_dict["kernel_counter"] = kernel_counter
                kernel_counter = kernel_counter + 1

                # Map the ops to numbers

                (reads, writes, total_bytes), nodes, node_read_writes, src_code = tuple(
                    pickle.load(open(pkl_path, "rb"))
                )
                op_counts = node_read_writes.op_counts
                op_bag = dict()
                for op in sorted(op_counts.keys()):
                    assert op in op_dict
                    op_bag[op_dict[op]] = op_counts[op]
                cur_kernel_dict["model"] = model
                cur_kernel_dict["kernel_name"] = kernel_name
                cur_kernel_dict["kernel_category"] = kernel_category
                cur_kernel_dict["num_of_loops"] = num_of_loops
                cur_kernel_dict["op_bag"] = op_bag

                # Get the size hints from src code
                size_hints = get_size_hints(model, kernel_name, src_code)

                # Get the stride and shape vec from flattened read_writes
                # sort the reads/writes according to the names of buf
                cur_kernel_dict["size_hints"] = size_hints
                sizevar_allocator = SizeVarAllocator()

                def f(rw_list):
                    res_list = list()
                    for dep, bytes in rw_list:
                        dep_dict = dict()
                        if isinstance(dep, (StarDep, WeakDep)):
                            dep_dict["StarDepOrWeakDep"] = True
                            continue
                        else:
                            dep_dict["StarDepOrWeakDep"] = False
                        dep_dict["bytes"] = bytes
                        strides = sizevar_allocator.stride_hints(
                            dep.index, dep.var_names
                        )
                        dep_dict["strides"] = strides
                        dep_dict["size"] = [int(size_) for size_ in dep.size]
                        dep_dict["is_contiguous"] = dep.is_contiguous()
                        dep_dict["is_scalar"] = dep.is_scalar()
                        dep_dict["is_indirect"] = dep.is_indirect()
                        dep_dict["name"] = dep.name

                        assert len(dep.size) == len(strides)
                        for size_ in dep.size:
                            assert size_.is_integer
                        for stride in strides:
                            assert isinstance(stride, int)
                        res_list.append(dep_dict)
                    return res_list

                cur_kernel_dict["reads_list"] = f(
                    sorted(zip(reads, total_bytes[: len(reads)]), key=lambda x: x[0][0])
                )
                cur_kernel_dict["writes_list"] = f(
                    sorted(
                        zip(writes, total_bytes[len(reads) :]), key=lambda x: x[0][0]
                    )
                )

                # Read all the configs
                with open(all_config_path, "r") as file:
                    all_config = json.load(file)
                    for config in all_config:
                        cur_kernel_dict["config"] = config
                        raw_data.append(cur_kernel_dict)
                        config_counter = config_counter + 1

with open("raw_data.pkl", "wb") as f:
    pickle.dump(raw_data, f)

print(op_dict)
print(kernel_counter)
print(config_counter)
