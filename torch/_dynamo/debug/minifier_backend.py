import copy
import dataclasses
import functools
import getpass
import logging
import os
import shutil
import subprocess
import textwrap
import uuid
from collections import Counter
from importlib import import_module
from tempfile import TemporaryFile

from functorch.compile import minifier

import torch
import torch.fx as fx
from torch._dynamo.eval_frame import lookup_backend
from torch._functorch.aot_autograd import make_boxed_func

from .. import config
from ..optimizations.backends import register_backend
from ..utils import clone_inputs, get_debug_dir
from torch._dynamo.utils import clone_inputs, same
from torch._subclasses import FakeTensorMode


from .serialize import *

log = logging.getLogger(__name__)


def minifier_dir():
    path = os.path.join(get_debug_dir(), "minifier")
    if path is None:
        path = f"/tmp/minifier_{getpass.getuser()}"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


@dataclasses.dataclass
class CompilationResult:
    compiled_fn: torch.fx.GraphModule
    already_minified: bool
    failed: bool
    reason: str


@dataclasses.dataclass
class RuntimeResult:
    out: list[torch.Tensor]
    already_minified: bool
    failed: bool
    reason: str


class MinifierException(Exception):
    def __init__(self, component, reason):
        self.component = component
        self.reason = reason
        self.message = f"Minifier saw a {component} issue. It will try to minify it. The original reason of failure is {reason}"
        super().__init__(self.message)


def run_fwd_maybe_bwd(gm, args, only_fwd=False):
    """
    Runs a forward and possibly backward iteration for a given mod and args.
    """
    import copy

    from torch._dynamo.testing import (
        collect_results,
        reduce_to_scalar_loss,
        requires_bwd_pass,
    )
    from torch._functorch.aot_autograd import make_boxed_func

    gm = copy.deepcopy(gm)
    new_args = clone_inputs(args)
    # Set the requires_grad field explicitly because clone_inputs only sets
    # requires_grad for leaf tensors.
    for narg, arg in zip(new_args, args):
        narg.requires_grad_(arg.requires_grad)
    args = new_args

    if hasattr(gm, "zero_grad"):
        gm.zero_grad(True)

    # TorchInductor returned callable expects lists. So, boxing the call.
    if not hasattr(gm, "_boxed_call") and hasattr(gm, "named_parameters"):
        orig_named_parameters = gm.named_parameters
        gm = make_boxed_func(gm)
        gm.named_parameters = orig_named_parameters

    out = gm(args)
    if only_fwd:
        return out
    if requires_bwd_pass(out):
        loss = reduce_to_scalar_loss(out)
        loss.backward()
    return collect_results(gm, out, None, args)


def compile_from_real_inputs(fx_g, compiler_fn, real_inputs):
    fake_mode = FakeTensorMode()
    fake_inputs = [fake_mode.from_tensor(x) for x in real_inputs]
    return compiler_fn(fx_g, fake_inputs)


def run_packed(model, real_inputs):
    return model(real_inputs)


def run_unpacked(model, real_inputs):
    return model(*real_inputs)


def raises_exception(fx_g, real_inputs, compiler_fn, compiled_fn, run_fn):
    if compiled_fn is None:
        compiled_fn = compile_from_real_inputs(fx_g, compiler_fn, real_inputs)

    try:
        out = run_fn(compiled_fn, real_inputs)
        return RuntimeResult(out, False, False, None)
    except MinifierException as exc:
        return RuntimeResult(None, True, False, exc)
    except Exception as exc:
        return RuntimeResult(None, False, True, exc)


def fails_accuracy(fx_g, real_inputs, compiler_fn, compiled_fn, run_ref_fn, run_fn):
    if compiled_fn is None:
        compiled_fn = compile_from_real_inputs(fx_g, compiler_fn, real_inputs)

    ref = run_ref_fn(fx_g, real_inputs)
    # ref = run_fwd_maybe_bwd(fx_g, real_inputs, True)
    try:
        # res = run_fwd_maybe_bwd(compiled_fn, real_inputs, True)
        cloned_inputs = clone_inputs(real_inputs)
        res = run_fn(compiled_fn, cloned_inputs)
    except MinifierException as exc:
        return RuntimeResult(None, True, False, exc)
    except Exception as exc:
        # Possible bad graph, just return ref
        return RuntimeResult(None, True, False, exc)
        # return RuntimeResult(ref, False, False, exc)

    if same(ref, res):
        return RuntimeResult(res, False, False, None)
    else:
        return RuntimeResult(None, False, True, "Accuracy failed")


def check_compilation(fx_g, compiler_fn, fake_inputs):
    try:
        compiled_fn = compiler_fn(fx_g, fake_inputs)
        return CompilationResult(compiled_fn, False, False, None)
    except MinifierException as exc:
        return CompilationResult(None, True, False, exc)
    except Exception as exc:
        return CompilationResult(None, False, True, exc)


def fails_runtime(fx_g, real_inputs, compiler_fn, compiled_fn, check_runtime_fn):
    return check_runtime_fn(fx_g, real_inputs, compiler_fn, compiled_fn).failed


def fails_compilation(fx_g, fake_inputs, compiler_fn, check_compilation_fn):
    return check_compilation_fn(fx_g, compiler_fn, fake_inputs).failed


def dump_graph(fx_g, inputs):
    repro_codegen = ReproInductorCodegen()
    repro = codegen_imports()
    repro += codegen_graph_module(fx_g)
    repro += codegen_inputs(inputs)
    repro += repro_codegen.codegen_module_instantiation()
    repro += repro_codegen.codegen_opt_mod_call()
    with open("repro.py", "w") as fw:
        fw.write(repro)


def run_minifier(gm, inputs, fails_fn, dump_fn):
    minifier(
        gm,
        inputs,
        module_fails=fails_fn,
        dump_state=dump_fn,
    )


def minifier_backend(
    fx_g,
    fake_inputs,
    compiler_fn,
    compiler_name,
    check_compilation_fn,
    check_runtime_fn,
    **kwargs,
):
    compiler_fn = lookup_backend(compiler_fn)
    compilation_result = check_compilation_fn(fx_g, compiler_fn, fake_inputs)
    if compilation_result.failed:
        print(
            f"Compilation failed at {compiler_name} {compiler_fn} with {compilation_result.reason}"
        )
        fails_compilation_fn = functools.partial(
            fails_compilation,
            compiler_fn=compiler_fn,
            check_compilation_fn=check_compilation_fn,
        )
        run_minifier(fx_g, fake_inputs, fails_compilation_fn, dump_graph)
        raise MinifierException("compiler", compilation_result.reason)
    elif compilation_result.already_minified:
        raise compilation_result.reason

    compiled_fn = compilation_result.compiled_fn
    print("Compilation successed")

    def runtime(*real_inputs):
        # TODO - This is hack because we are sharing runtime function between inductor and runtime.
        # Separate it out.
        if compiler_name == "inductor_inner":
            real_inputs = real_inputs[0]
        print(compiler_name)
        cloned_inputs = clone_inputs(real_inputs)
        runtime_result = check_runtime_fn(fx_g, real_inputs, compiler_fn, compiled_fn)
        print(runtime_result)
        print(runtime_result.failed)
        if runtime_result.failed:
            print(f"Runtime failed with {runtime_result.reason}", flush=True)
            # Note that we are setting compiled_fn to None, so that minifier recompiles
            fails_runtime_fn = functools.partial(
                fails_runtime,
                compiler_fn=compiler_fn,
                check_runtime_fn=check_runtime_fn,
                compiled_fn=None,
            )
            run_minifier(fx_g, cloned_inputs, fails_runtime_fn, dump_graph)
            raise MinifierException("runtime", runtime_result.reason)
        elif runtime_result.already_minified:
            raise runtime_result.reason
        return runtime_result.out

    # TODO - Hack because runtime is defined here. We should keep this function clean.
    if compiler_name == "inductor_inner":
        # This is needed
        runtime._boxed_call = True
    return runtime


def debug_wrapper(compiler_fn, compiler_name):
    @functools.wraps(compiler_fn)
    def inner(gm, example_inputs, **kwargs):
        nonlocal compiler_name

        # TODO - Can we clean this up?
        if compiler_name != "inductor_inner":
            run_fn = run_unpacked
        else:
            run_fn = run_packed

        if (
            config.repro_after == "dynamo"
            and compiler_name != "inductor_inner"
            or config.repro_after == "aot"
            and compiler_name == "inductor_inner"
        ):
            check_compilation_fn = check_compilation
            if config.repro_level in (1, 2, 3):
                check_runtime_fn = functools.partial(raises_exception, run_fn=run_fn)
            elif config.repro_level == 4:
                # check_runtime_fn = functools.partial(raises_exception, run_fn=run_fn)
                check_runtime_fn = functools.partial(
                    fails_accuracy, run_ref_fn=run_unpacked, run_fn=run_fn
                )
            # TODO - compiler_fn, compiler_fn
            compiled_obj = minifier_backend(
                gm,
                example_inputs,
                compiler_fn,
                compiler_name,
                check_compilation_fn,
                check_runtime_fn,
                **kwargs,
            )
        else:
            compiled_obj = compiler_fn(gm, example_inputs, **kwargs)

        return compiled_obj

    return inner
