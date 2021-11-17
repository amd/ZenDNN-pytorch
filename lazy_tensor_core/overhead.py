
import argparse
import collections
import copy
import csv
import functools
import gc
import io
import logging
import math
import numpy as np
import os
import re
import sys
import textwrap
import time
import torch
import torch.nn.functional as F
import torch.autograd.profiler as profiler
import warnings
from torch import nn
from torch.nn import Module
from os.path import abspath
from os.path import exists
from scipy.stats import gmean
from scipy.stats import ttest_ind

# from caffe2.python import workspace
# workspace.GlobalInit(['caffe2', '--caffe2_log_level=-5'])

import lazy_tensor_core
import lazy_tensor_core.core.lazy_model as ltm
import lazy_tensor_core.debug.metrics as met
lazy_tensor_core._LAZYC._ltc_init_ts_backend()


os.environ["KALDI_ROOT"] = "/tmp"  # avoids some spam
torchbench_dir = abspath("/home/whc/benchmark")
assert os.path.exists(torchbench_dir)
# os.chdir(torchbench_dir)
sys.path.append(torchbench_dir)
log = logging.getLogger(__name__)
SKIP = {}
current_name = ""
current_device = ""

def synchronize():
    pass

@functools.lru_cache(1)
def output_csv(name, headers):
    output = csv.writer(
        io.TextIOWrapper(
            open(name, "wb", buffering=0),
            "utf-8",
            write_through=True,
        )
    )
    output.writerow(headers)
    return output


class Fusion(nn.Module):
    def __init__(self, dims=[128, 16, 128, 128], dev='cuda'):
        super(Fusion, self).__init__()
        self.attention_head_size = dims[1]
        self.example_inputs = (
            torch.randn(*dims, device=dev, dtype=torch.float32),
            torch.randn(*dims, device=dev, dtype=torch.float32),
        )

    def get_module(self):
        return self, self.example_inputs

    def forward(self, inputs, mask):
        out1 = inputs / math.sqrt(self.attention_head_size)
        out2 = out1 + mask
        out3 = out2 * 5.0
        return out3

def pick_grad(name):
    if name in ("maml",):
        return torch.enable_grad()
    else:
        return torch.no_grad()

def short_name(name, limit=20):
    """Truncate a model name to limit chars"""
    return name if len(name) <= limit else f"{name[:limit - 3].rstrip('_')}..."


# Iter torchbench models
def iter_models(args):
    from fastNLP.core import logger

    logger.setLevel(logging.WARNING)
    from torchbenchmark import list_models  # noqa
    for benchmark_cls in list_models():
        if (
            (len(args.filter) and (not re.search("|".join(args.filter), benchmark_cls.name, re.I)))
            or (len(args.exclude) and re.search("|".join(args.exclude), benchmark_cls.name, re.I))
            or benchmark_cls.name in SKIP
        ):
            continue
        for device in args.devices:
            try:
                benchmark = benchmark_cls(device=device, jit=False)
                lazy_benchmark = benchmark_cls(device='lazy', jit=False)
                model, example_inputs = benchmark.get_module()
                lazy_model, lazy_example_inputs = lazy_benchmark.get_module()
                model.eval()
                lazy_model.eval()
                gc.collect()
                global current_name, current_device
                current_device = device
                current_name = short_name(benchmark.name)
                yield device, current_name, model, example_inputs, lazy_model, lazy_example_inputs
            except NotImplementedError:
                print("NotImplementedError")
                pass
            except Exception as e:
                print(f"Exception in iter_models for {benchmark_cls.name}", e)
                log.exception(f"misconfigured model {benchmark_cls.name}")

def call_model_with(model, inputs):
    if isinstance(inputs, tuple) or isinstance(inputs, list):
        return model(*inputs)
    elif isinstance(inputs, dict):
        return model(**inputs)
    elif isistance(inputs, torch.Tensor):
        return model(inputs)
    raise RuntimeError("invalid example inputs ", inputs)

def timed(model, example_inputs, times=1):
    
    synchronize() # noop currenly
    gc.collect()
    torch.manual_seed(1337)
    t0 = time.perf_counter()
    for _ in range(times):
        result = call_model_with(model, example_inputs)
        synchronize()
    t1 = time.perf_counter()
    return result, t1 - t0

def example_inputs_to(inputs, device):
    if isinstance(inputs, tuple):
        return tuple(i.to(device) for i in inputs)
    elif isinstance(inputs, dict):
        return {k: inputs[k].to(device) for k in inputs}
    elif isistance(inputs, torch.Tensor):
        return inputs.to(device)
    raise RuntimeError("invalid example inputs ", inputs)

def lazy_overhead_experiment(results, args, model, example_inputs, lazy_model, lazy_inputs):
    timings = np.zeros((args.repeat, 2), np.float64)
    # lazy_inputs = example_inputs_to(example_inputs, 'lazy')
    # lazy_model = model.to('lazy')
    for rep in range(args.repeat):
        # interleave the runs to handle frequency scaling and load changes
        _, timings[rep, 0] = timed(model, example_inputs)
        _, timings[rep, 1] = timed(lazy_model, lazy_inputs)
    pvalue = ttest_ind(timings[:, 0], timings[:, 1]).pvalue
    median = np.median(timings, axis=0)
    overhead = median[1] / median[0]
    results.append(overhead)
    output_csv(
        "lazy_overheads.csv",
        ("dev", "name", "overhead"),
    ).writerow([current_device, current_name, f"{overhead:.4f}"])
    return (overhead, pvalue)

def check_results(name, correct_result, lazy_result, device):
    import transformers #noqa
    if isinstance(correct_result, transformers.modeling_outputs.MaskedLMOutput):
        correct_result = correct_result.to_tuple()[0]
        lazy_result = lazy_result.to_tuple()[0]
    lazy_result = lazy_result.to(device)
    return torch.allclose(correct_result, lazy_result)


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", "-k", action="append", default=["hf_Bert"], help="filter benchmarks")
    parser.add_argument("--exclude", "-x", action="append", default=[], help="filter benchmarks")
    parser.add_argument("--devices", "-d", action="append", default=['cuda'], help="cpu or cuda")
    parser.add_argument(
        "--repeat", "-n", type=int, default=40, help="number of timing runs"
    )
    args = parser.parse_args()
    results = []

    # behave more like a torchbenchmark (get_module) so the tools are reusable
    benchmark = Fusion()
    model, example_inputs = benchmark.get_module()
    lazy_benchmark = Fusion(dev='lazy')
    lazy_model, lazy_inputs = lazy_benchmark.get_module()
    overhead, pvalue = lazy_overhead_experiment(results, args, model, example_inputs, lazy_model, lazy_inputs)
    print(f"name: Fusion, overhead: {overhead}, pvalue: {pvalue}")

    for device, name, model, example_inputs, lazy_model, lazy_inputs in iter_models(args):
        if device == 'cuda':
            assert 'LTC_TS_CUDA' in os.environ and bool(os.environ['LTC_TS_CUDA'])

        with pick_grad(name):
            try:
                torch.manual_seed(1337)
                correct_result = call_model_with(copy.deepcopy(model), example_inputs)
                torch.manual_seed(1337)
                lazy_result = call_model_with(lazy_model, lazy_inputs)
            except Exception:
                logging.exception("unhandled error")
                print("ERROR")
                continue
            if not check_results(name, correct_result, lazy_result, device):
                print("INCORRECT")
                continue
            overhead, pvalue = lazy_overhead_experiment(results, args, model, example_inputs, lazy_model, lazy_inputs)
            print(f"name: {name},  overhead: {overhead}, pvalue: {pvalue}")
        
    """
for lazy tensor:
t1-t0  0.0006442070007324219 , t2-t1  0.00501251220703125
t1-t0  0.0002551078796386719 , t2-t1  0.044301748275756836
t1-t0  0.00029778480529785156 , t2-t1  0.028578758239746094
t1-t0  0.00030732154846191406 , t2-t1  0.03235149383544922
t1-t0  0.0003490447998046875 , t2-t1  0.037885189056396484
t1-t0  0.0006480216979980469 , t2-t1  0.03445005416870117
t1-t0  0.000667572021484375 , t2-t1  0.03567194938659668
t1-t0  0.0006542205810546875 , t2-t1  0.04100847244262695
t1-t0  0.0006890296936035156 , t2-t1  0.05210447311401367
t1-t0  0.0006821155548095703 , t2-t1  0.049530744552612305
t1-t0  0.0006682872772216797 , t2-t1  0.055863380432128906
t1-t0  0.00038552284240722656 , t2-t1  0.04713630676269531
t1-t0  0.0008995532989501953 , t2-t1  0.04043078422546387
t1-t0  0.00045752525329589844 , t2-t1  0.037316322326660156
t1-t0  0.00039768218994140625 , t2-t1  0.0435786247253418
t1-t0  0.0009481906890869141 , t2-t1  0.03600001335144043
t1-t0  0.0005481243133544922 , t2-t1  0.04508638381958008
t1-t0  0.0004177093505859375 , t2-t1  0.049938201904296875
t1-t0  0.0008950233459472656 , t2-t1  0.04341864585876465
t1-t0  0.0006542205810546875 , t2-t1  0.043274879455566406
t3-t2  0.0441899299621582
    """

    
    """ with mul codegenned, and non-dbg build
    (.venv) [whc@fedora lazy_tensor_core]$ LTC_TS_CUDA=1 python overhead.py t1-t0  0.00024127960205078125 , t2-t1  0.003977060317993164
t1-t0  7.43865966796875e-05 , t2-t1  0.010112762451171875
t1-t0  6.031990051269531e-05 , t2-t1  0.0007240772247314453
t1-t0  5.6743621826171875e-05 , t2-t1  0.17930293083190918
t1-t0  0.0002052783966064453 , t2-t1  0.00013518333435058594
t1-t0  9.202957153320312e-05 , t2-t1  0.00027871131896972656
t1-t0  0.00010824203491210938 , t2-t1  0.0002682209014892578
t1-t0  8.082389831542969e-05 , t2-t1  0.0002703666687011719
t1-t0  8.058547973632812e-05 , t2-t1  0.0001933574676513672
t1-t0  8.869171142578125e-05 , t2-t1  0.00015115737915039062
t1-t0  7.963180541992188e-05 , t2-t1  0.00026679039001464844
t1-t0  7.772445678710938e-05 , t2-t1  0.00010538101196289062
t1-t0  8.20159912109375e-05 , t2-t1  8.130073547363281e-05
t1-t0  8.344650268554688e-05 , t2-t1  0.0002009868621826172
t1-t0  7.915496826171875e-05 , t2-t1  7.915496826171875e-05
t1-t0  7.748603820800781e-05 , t2-t1  7.939338684082031e-05
t1-t0  7.581710815429688e-05 , t2-t1  0.0001049041748046875
t1-t0  8.130073547363281e-05 , t2-t1  7.939338684082031e-05
t1-t0  7.486343383789062e-05 , t2-t1  0.00015282630920410156
t1-t0  7.748603820800781e-05 , t2-t1  0.0001399517059326172
t3-t2  8.893013000488281e-05
Metric: DeviceLockWait

without mul codegenned, similar times

in debug mode - causes 3-5X slowdown in tracing overhead:
t1-t0  0.0004985332489013672 , t2-t1  0.0047452449798583984
t1-t0  0.00022363662719726562 , t2-t1  0.01014852523803711
t1-t0  0.00017976760864257812 , t2-t1  0.0008509159088134766
t1-t0  0.00014472007751464844 , t2-t1  0.18119335174560547
t1-t0  0.0006041526794433594 , t2-t1  0.00039196014404296875
t1-t0  0.0004391670227050781 , t2-t1  0.00032639503479003906
t1-t0  0.0004451274871826172 , t2-t1  0.0003170967102050781
t1-t0  0.0004448890686035156 , t2-t1  0.00032210350036621094
t1-t0  0.00041604042053222656 , t2-t1  0.00032806396484375
t1-t0  0.0004172325134277344 , t2-t1  0.00031828880310058594
t1-t0  0.00042891502380371094 , t2-t1  0.0003159046173095703
t1-t0  0.0004267692565917969 , t2-t1  0.0003170967102050781
t1-t0  0.0004146099090576172 , t2-t1  0.00032591819763183594
t1-t0  0.00041794776916503906 , t2-t1  0.0003135204315185547
t1-t0  0.0004284381866455078 , t2-t1  0.0003132820129394531
t1-t0  0.0004134178161621094 , t2-t1  0.00034546852111816406
t1-t0  0.0004189014434814453 , t2-t1  0.00032520294189453125
t1-t0  0.0004143714904785156 , t2-t1  0.00031447410583496094
t1-t0  0.0004279613494873047 , t2-t1  0.0003132820129394531
t1-t0  0.0004127025604248047 , t2-t1  0.00032448768615722656
t3-t2  9.918212890625e-05
    """