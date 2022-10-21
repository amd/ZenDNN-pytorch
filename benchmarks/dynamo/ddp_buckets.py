import argparse
import copy
import tabulate
import torch

import torch._dynamo as dynamo
import torch.utils._pytree as pytree
from torch._dynamo.optimizations import BACKENDS
from torch._dynamo.optimizations.distributed import DDPOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    from .dist_util import model_iter_fn
except ImportError:
    from dist_util import model_iter_fn




def print_ddp_buckets(args, model, inputs):
    def move_tensor(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda()
        return maybe_tensor

    inputs = pytree.tree_map(move_tensor, inputs)
    model = model.cuda()
    ddp_model = DDP(copy.deepcopy(model))
    # warmup
    for _ in range(3):
        model_iter_fn(ddp_model, inputs, collect_outputs=False)
    buckets = ddp_model.reducer._get_zeros_like_grad_buckets()
    assert all([b.buffer().dim() == 1 for b in buckets])
    ddp_buckets = [int(b.buffer().storage().nbytes()) for b in buckets]

    # build our own ddp-optimizer so we can get its internal state- so don't double-optimize
    dynamo.config.optimize_ddp = False
    ddp_opt = DDPOptimizer(
        ddp_model.bucket_bytes_cap,
        parameters_to_ignore=[],
        backend_compile_fn=BACKENDS["aot_eager"],
        debug=True,
    )
    dynamo_ctx = dynamo.optimize(ddp_opt.compile_fn)
    # don't reuse ddp_model since we want to ensure we're not changing the behavior of dynamo+ddp
    dynamo_model = dynamo_ctx(DDP(copy.deepcopy(model)))
    for _ in range(1):
        model_iter_fn(dynamo_model, inputs, collect_outputs=False)

    # opt_buckets = list(reversed(ddp_opt.bucket_actual_sizes))
    
    # opt_names = "\n".join(map(str, ddp_opt.bucket_param_names))
    opt_names = ""  # todo
    headers = ("index", "DDP sz", "DDP-Opt sz", "Status", "DDP-Opt params")
    rows = []
    n_buckets = len(ddp_buckets)
    for i in range(n_buckets):
        opt = opt_buckets[i] if i < len(opt_buckets) else None
        mismatch = "error" if opt != ddp_buckets[i] else ""
        rows.append([i, ddp_buckets[i], opt, mismatch, opt_names])
    for i, opt in enumerate(opt_buckets[n_buckets:]):
        rows.append([i, "", opt, "!!!", ""])

    rows.append([])
    s_d = sum(ddp_buckets)
    s_o = sum(opt_buckets)
    rows.append(["SUM", s_d, s_o, "error" if s_d != s_o else None, None])

    print(tabulate.tabulate(rows, headers=headers, tablefmt="rounded_grid"))
    print(
        "Buckets printed in order of execution (0 first, corresponding to last output layers of fwd)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", default=None)
    model_arg = parser.add_mutually_exclusive_group(required=True)
    model_arg.add_argument(
        "--torchbench_model", help="name of torchbench model, e.g. hf_Bert"
    )
    model_arg.add_argument(
        "--toy_model", action="store_true", help="use toy model instead"
    )
    args = parser.parse_args()
    model_name = "ToyModel" if args.toy_model else args.torchbench_model
    model, inputs = get_model(args)


    setup(0, 1)
    print_ddp_buckets(args, model, inputs)
    cleanup()
