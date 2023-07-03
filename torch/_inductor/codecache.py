import base64
import dataclasses
import functools
import getpass
import hashlib
import importlib
import json
import logging
import multiprocessing
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
import threading
import types
import weakref
from bisect import bisect_right
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from ctypes import cdll
from dataclasses import field
from functools import partial
from importlib import abc
from threading import Thread
from time import sleep, time
from typing import Any, Callable, Dict, List, Set, Union

import torch

from torch._inductor import config, cuda_properties, exc
from torch._inductor.utils import developer_warning
from torch.hub import _Faketqdm, tqdm

_HERE = os.path.abspath(__file__)
_TORCH_PATH = os.path.dirname(os.path.dirname(_HERE))

if config.is_fbcode():
    from triton.fb import build_paths
    from triton.fb.build import _run_build_command

    from torch._inductor.fb.logging import global_cache_log
else:

    def global_cache_log(*args, **kwargs):
        pass


LOCK_TIMEOUT = 600

# timing metrics for time spent in the compilation
_cumulative_compile_time = 0
_t0 = None


def _compile_start():
    global _t0
    if _t0 is None:
        _t0 = time()


def _compile_end():
    global _cumulative_compile_time, _t0
    if _t0 is not None:
        t1 = time()
        _cumulative_compile_time += t1 - _t0
        _t0 = None
        # print("CUMULATIVE COMPILE TIME", _cumulative_compile_time)


log = logging.getLogger(__name__)


@functools.lru_cache(None)
def cache_dir():
    cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if cache_dir is None:
        cache_dir = f"{tempfile.gettempdir()}/torchinductor_{getpass.getuser()}"
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def cpp_wrapper_cache_dir(name):
    cu_str = (
        "cpu"
        if torch.version.cuda is None
        else f'cu{torch.version.cuda.replace(".", "")}'
    )
    python_version = f"py{sys.version_info.major}{sys.version_info.minor}"
    build_folder = f"{python_version}_{cu_str}"

    cpp_wrapper_dir = os.path.join(cache_dir(), build_folder)
    cpp_wrapper_build_directory = os.path.join(cpp_wrapper_dir, name)
    os.makedirs(cpp_wrapper_build_directory, exist_ok=True)
    return cpp_wrapper_build_directory


class CacheBase:
    @staticmethod
    @functools.lru_cache(None)
    def get_system():
        try:
            import triton

            triton_version = triton.__version__
        except ModuleNotFoundError:
            triton_version = None

        system = {
            "device": {
                "name": torch.cuda.get_device_properties(
                    torch.cuda.current_device()
                ).name,
            },
            "version": {
                "cuda": torch.version.cuda,
                "triton": triton_version,
            },
        }

        system["hash"] = hashlib.sha256(
            json.dumps(system, sort_keys=True).encode("utf-8")
        ).hexdigest()

        return system

    def __init__(self):
        if not torch.cuda.is_available():
            return

        self.system = CacheBase.get_system()

        self.local_cache_path = os.path.join(cache_dir(), self.system["hash"])
        self.global_cache_path = (
            os.path.join(os.path.dirname(config.global_cache_dir), self.system["hash"])
            if config.global_cache_dir is not None
            else None
        )

    def get_local_cache(self):
        if not os.path.isfile(self.local_cache_path):
            return {}
        with open(self.local_cache_path, "r") as local_cache_fp:
            local_cache = json.load(local_cache_fp)
        return local_cache["cache"]

    def update_local_cache(self, local_cache):
        write_atomic(
            self.local_cache_path,
            json.dumps({"system": self.system, "cache": local_cache}, indent=4),
        )


class LocalCache(CacheBase):
    def lookup(self, *keys: List[str]):
        cache = self.get_local_cache()

        sub_cache = cache
        for key in keys:
            if key in cache:
                sub_cache = cache[key]
            else:
                return None

        return sub_cache

    def set_value(self, *keys: List[str], value: Any):
        cache = self.get_local_cache()

        sub_cache = cache
        for key in keys[0:-1]:
            sub_cache.setdefault(key, {})
            sub_cache = sub_cache[key]
        sub_cache[keys[-1]] = value

        self.update_local_cache(cache)


class PersistentCache(CacheBase):
    @functools.lru_cache(None)
    def get_global_cache(self):
        if self.global_cache_path is None or not os.path.isfile(self.global_cache_path):
            return {}
        with open(self.global_cache_path, "r") as global_cache_fp:
            global_cache = json.load(global_cache_fp)
        return global_cache["cache"]

    def lookup(
        self,
        choices,
        name: str,
        inputs: str,
        benchmark: Callable[[Any], float],
    ):
        """
        Check to see if we have benchmarked the given choice callers. For each
        choice caller:

            1. Check global_cache[name][inputs][choice], return benchmark if cached.
            2. Check local_cache[name][inputs][choice], return benchmark if cached.
            3.
                a. `max_autotune_gemm=True`: benchmark the choice, update
                    local_cache[name][inputs][choice], and return the benchmark.
                b. `max_autotune_gemm=False`: don't benchmark the choice, return nothing.
        """

        gc_log = partial(global_cache_log, self.system, name, inputs)
        timings = {}

        def check_cache(cache, callback=None):
            """Check if `cache` contains data for all the choices"""
            hit = True
            for choice in choices:
                choice_hash = choice.hash_key()
                if choice_hash in cache.get(name, {}).get(inputs, {}):
                    # cache hit
                    timings[choice] = cache[name][inputs][choice_hash]
                    if callback:
                        callback(choice_hash, cached=True)
                else:
                    # cache miss
                    hit = False
                    if callback:
                        callback(choice_hash, cached=False)
            return hit

        if config.max_autotune or config.max_autotune_gemm:
            local_cache = self.get_local_cache()
            # check local cache first since it is data specific to the current machine
            if not check_cache(local_cache) and not check_cache(
                self.get_global_cache(), callback=gc_log
            ):
                # re-benchmark everything to try to get consistent numbers from the same machine
                for choice in choices:
                    timings[choice] = benchmark(choice)
                    local_cache.setdefault(name, {})
                    local_cache[name].setdefault(inputs, {})
                    local_cache[name][inputs][choice.hash_key()] = timings[choice]

                self.update_local_cache(local_cache)
        else:
            # only check global cache, not local one
            check_cache(self.get_global_cache(), callback=gc_log)
            # may have a partial cache hit, where not everything is benchmarked

        return timings


def get_lock_dir():
    lock_dir = os.path.join(cache_dir(), "locks")
    if not os.path.exists(lock_dir):
        os.makedirs(lock_dir, exist_ok=True)
    return lock_dir


def code_hash(code, extra=""):
    hashing_str = code
    if extra != "":
        hashing_str = hashing_str + "||" + extra
    return (
        "c"
        + base64.b32encode(hashlib.sha256(hashing_str.encode("utf-8")).digest())[:51]
        .decode("utf-8")
        .lower()
    )


def get_path(basename: str, extension: str):
    subdir = os.path.join(cache_dir(), basename[1:3])
    path = os.path.join(subdir, f"{basename}.{extension}")
    return basename, subdir, path


def get_hash(content: Union[str, bytes], extra="", hash_type="code"):
    assert hash_type in ["code", "cubin"], "Hash type not supported"
    if hash_type == "code":
        return code_hash(content, extra)
    if hash_type == "cubin":
        return code_hash(repr(content))


def write(
    content: Union[str, bytes], extension: str, extra="", hash_type: str = "code"
):
    key: str = get_hash(content, extra, hash_type)
    basename, subdir, path = get_path(key, extension)
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    if not os.path.exists(path):
        write_atomic(path, content)
    return basename, path


def write_atomic(path: str, content: Union[str, bytes]):
    # Write into temporary file first to avoid conflicts between threads
    # Avoid using a named temporary file, as those have restricted permissions
    assert isinstance(
        content, (str, bytes)
    ), "Only strings and byte arrays can be saved in the cache"
    path = pathlib.Path(path)
    tmp_path = path.parent / f".{os.getpid()}.{threading.get_ident()}.tmp"
    write_mode = "w" if isinstance(content, str) else "wb"
    with tmp_path.open(write_mode) as f:
        f.write(content)
    tmp_path.rename(path)


@dataclasses.dataclass
class CompiledFxGraph:
    """Class holding a compiled FX graph"""

    compiled_artifact: Callable = None
    current_callable: Callable = None
    cache_key: str = None
    artifact_path: str = None
    cache_linemap: List = None
    device_types: Set[str] = field(default_factory=set)
    device_idxs: Set[int] = field(default_factory=set)
    mutated_inputs: Set[str] = field(default_factory=set)
    _boxed_call: bool = None

    def __call__(self, inputs) -> Any:
        return self.get_current_callable()(inputs)

    def get_current_callable(self):
        if self.current_callable is None:
            # This prevents a circular reference that makes CompiledFxGraph
            # get stuck without getting garbage collected
            return functools.partial(_run_from_cache, weakref.proxy(self))
        else:
            return self.current_callable


def _run_from_cache(compiled_graph: CompiledFxGraph, inputs):
    # We can't really serialize callables that may be C++/Triton/etc.,
    # so we serialize their disk cache location instead
    # TODO: When making an API that can save compiled models e2e to disk
    # this will need to be better
    if compiled_graph.compiled_artifact is None:
        from .codecache import PyCodeCache

        compiled_graph.compiled_artifact = PyCodeCache.load_by_key_path(
            compiled_graph.cache_key,
            compiled_graph.artifact_path,
            compiled_graph.cache_linemap
            if compiled_graph.cache_linemap is not None
            else (),
        ).call

    return compiled_graph.compiled_artifact(inputs)


def cpp_compiler():
    if config.is_fbcode():
        return build_paths.gcc()
    if isinstance(config.cpp.cxx, (list, tuple)):
        search = tuple(config.cpp.cxx)
    else:
        search = (config.cpp.cxx,)
    return cpp_compiler_search(search)


@functools.lru_cache(1)
def cpp_compiler_search(search):
    for cxx in search:
        try:
            if cxx is None:
                # gxx package is only available for Linux
                # according to https://anaconda.org/conda-forge/gxx/
                if sys.platform != "linux":
                    continue
                # Do not install GXX by default
                if not os.getenv("TORCH_INDUCTOR_INSTALL_GXX"):
                    continue
                from filelock import FileLock

                lock_dir = get_lock_dir()
                lock = FileLock(
                    os.path.join(lock_dir, "g++.lock"), timeout=LOCK_TIMEOUT
                )
                with lock:
                    cxx = install_gcc_via_conda()
            subprocess.check_output([cxx, "--version"])
            return cxx
        except (subprocess.SubprocessError, FileNotFoundError, ImportError):
            continue
    raise exc.InvalidCxxCompiler()


def install_gcc_via_conda():
    """On older systems, this is a quick way to get a modern compiler"""
    prefix = os.path.join(cache_dir(), "gcc")
    cxx_path = os.path.join(prefix, "bin", "g++")
    if not os.path.exists(cxx_path):
        log.info("Downloading GCC via conda")
        conda = os.environ.get("CONDA_EXE", "conda")
        if conda is None:
            conda = shutil.which("conda")
        if conda is not None:
            subprocess.check_call(
                [
                    conda,
                    "create",
                    f"--prefix={prefix}",
                    "--channel=conda-forge",
                    "--quiet",
                    "-y",
                    "python=3.8",
                    "gxx",
                ],
                stdout=subprocess.PIPE,
            )
    return cxx_path


def is_gcc():
    return re.search(r"(gcc|g\+\+)", cpp_compiler())


class VecISA:
    _bit_width: int
    _macro: str
    _arch_flags: str
    _dtype_nelements: Dict[torch.dtype, int]

    # Note [Checking for Vectorized Support in Inductor]
    # TorchInductor CPU vectorization reuses PyTorch vectorization utility functions
    # Hence, TorchInductor would depend on Sleef* to accelerate mathematical functions
    # like exp, pow, sin, cos and etc.
    # But PyTorch and TorchInductor might use different compilers to build code. If
    # PyTorch uses gcc-7/g++-7 to build the release package, the libtorch_cpu.so
    # will not expose the Sleef* AVX512 symbols since gcc-7/g++-7 cannot pass
    # avx512 check in CMake - FindAVX.cmake. But TorchInductor install the latest
    # gcc/g++ compiler by default while it could support the AVX512 compilation.
    # Therefore, there would be a conflict sleef version between PyTorch and
    # TorchInductor. Hence, we dry-compile the following code to check whether current
    # HW platform and PyTorch both could support AVX512 or AVX2. And suppose ARM
    # also needs the logic
    # In fbcode however, we are using the same compiler for pytorch and for inductor codegen,
    # making the runtime check unnecessary.
    _avx_code = """
#if defined(CPU_CAPABILITY_AVX512) || defined(CPU_CAPABILITY_AVX2)
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#endif

__attribute__((aligned(64))) float in_out_ptr0[16] = {0.0};

extern "C" void __avx_chk_kernel() {
    auto tmp0 = at::vec::Vectorized<float>(1);
    auto tmp1 = tmp0.exp();
    tmp1.store(in_out_ptr0);
}
"""

    _avx_py_load = """
import torch
from ctypes import cdll
cdll.LoadLibrary("__lib_path__")
"""

    def bit_width(self):
        return self._bit_width

    def nelements(self, dtype: torch.dtype = torch.float):
        return self._dtype_nelements[dtype]

    def build_macro(self):
        return self._macro

    def build_arch_flags(self):
        return self._arch_flags

    def __hash__(self) -> int:
        return hash(str(self))

    @functools.lru_cache(None)
    def __bool__(self):
        if config.cpp.vec_isa_ok is not None:
            return config.cpp.vec_isa_ok

        key, input_path = write(VecISA._avx_code, "cpp")
        from filelock import FileLock

        lock_dir = get_lock_dir()
        lock = FileLock(os.path.join(lock_dir, key + ".lock"), timeout=LOCK_TIMEOUT)
        with lock:
            output_path = input_path[:-3] + "so"
            build_cmd = cpp_compile_command(
                input_path, output_path, warning_all=False, vec_isa=self
            ).split(" ")
            try:
                # Check build result
                compile_file(input_path, output_path, build_cmd)
                # Note [CPU vectorization in fbcode]
                # TODO: get vectorization working in fbcode.
                # For now, this always fails, so we fall back to generating non-vectorized cpu code.
                subprocess.check_call(
                    [
                        "python",
                        "-c",
                        VecISA._avx_py_load.replace("__lib_path__", output_path),
                    ],
                    stderr=subprocess.DEVNULL,
                )
            except Exception as e:
                return False

            return True


@dataclasses.dataclass
class VecAVX512(VecISA):
    _bit_width = 512
    _macro = "CPU_CAPABILITY_AVX512"
    _arch_flags = "-mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma"
    _dtype_nelements = {torch.float: 16, torch.bfloat16: 32}

    def __str__(self) -> str:
        return "avx512"

    __hash__: Callable[[VecISA], Any] = VecISA.__hash__


@dataclasses.dataclass
class VecAVX2(VecISA):
    _bit_width = 256
    _macro = "CPU_CAPABILITY_AVX2"
    _arch_flags = "-mavx2 -mfma"
    _dtype_nelements = {torch.float: 8, torch.bfloat16: 16}

    def __str__(self) -> str:
        return "avx2"

    __hash__: Callable[[VecISA], Any] = VecISA.__hash__


class InvalidVecISA(VecISA):
    _bit_width = 0
    _macro = ""
    _arch_flags = ""
    _dtype_nelements = {}

    def __str__(self) -> str:
        return "INVALID_VEC_ISA"

    def __bool__(self):
        return False

    __hash__: Callable[[VecISA], Any] = VecISA.__hash__


invalid_vec_isa = InvalidVecISA()
supported_vec_isa_list = [VecAVX512(), VecAVX2()]


# Cache the cpuinfo to avoid I/O overhead. Meanwhile, the cpuinfo content
# might have too much redundant content that is useless for ISA check. Hence,
# we only cache some key isa information.
@functools.lru_cache(None)
def valid_vec_isa_list():
    if sys.platform != "linux":
        return []

    isa_list = []
    with open("/proc/cpuinfo") as _cpu_info:
        _cpu_info_content = _cpu_info.read()
        for isa in supported_vec_isa_list:
            if str(isa) in _cpu_info_content and isa:
                isa_list.append(isa)
        return isa_list


def pick_vec_isa():
    # See Note [CPU vectorization in fbcode]
    # (this call hangs in fbcode)
    _valid_vec_isa_list: List[VecISA] = [] if config.is_fbcode() else valid_vec_isa_list()
    if not _valid_vec_isa_list:
        return invalid_vec_isa

    # If the simdlen is None, it indicates determin the vectorization length automatically
    if config.cpp.simdlen is None:
        assert _valid_vec_isa_list
        return _valid_vec_isa_list[0]

    for isa in _valid_vec_isa_list:
        if config.cpp.simdlen == isa.bit_width():
            return isa

    return invalid_vec_isa


def get_shared(shared=True):
    return "-shared -fPIC" if shared else ""


def get_warning_all_flag(warning_all=True):
    return "-Wall" if warning_all else ""


def cpp_flags():
    return "-std=c++17 -Wno-unused-variable"


def optimization_flags():
    base_flags = "-O3 -ffast-math -fno-finite-math-only"
    if config.is_fbcode():
        # FIXME: passing `-fopenmp` adds libgomp.so to the generated shared library's dependencies.
        # This causes `ldopen` to fail in fbcode, because libgomp does not exist in the default paths.
        # We will fix it later by exposing the lib path.
        return base_flags

    if sys.platform == "darwin":
        # Per https://mac.r-project.org/openmp/ right way to pass `openmp` flags to MacOS is via `-Xclang`
        # Also, `-march=native` is unrecognized option on M1
        base_flags += " -Xclang"
    else:
        base_flags += " -march=native"

    # Internal cannot find libgomp.so
    if not config.is_fbcode():
        base_flags += " -fopenmp"
    return base_flags


def use_custom_generated_macros():
    return "-D C10_USING_CUSTOM_GENERATED_MACROS"


def use_fb_internal_macros():
    if config.is_fbcode():
        return "-D C10_USE_GLOG -D C10_USE_MINIMAL_GLOG -D C10_MOBILE"
    else:
        return ""


def get_include_and_linking_paths(
    include_pytorch=False, vec_isa: VecISA = invalid_vec_isa, cuda=False, aot_mode=False
):
    from torch.utils import cpp_extension

    macros = ""
    if sys.platform == "linux" and (
        include_pytorch
        or vec_isa != invalid_vec_isa
        or cuda
        or config.cpp.enable_kernel_profile
    ):
        # Note - We include pytorch only on linux right now. There is more work
        # to do to enable OMP build on darwin where PyTorch is built with IOMP
        # and we need a way to link to what PyTorch links.
        ipaths = cpp_extension.include_paths(cuda) + [sysconfig.get_path("include")]
        if aot_mode:
            ipaths += [os.path.join(_TORCH_PATH, "_inductor", "aot_inductor_include")]
        lpaths = cpp_extension.library_paths(cuda) + [
            sysconfig.get_config_var("LIBDIR")
        ]
        libs = []
        # No need to manually specify libraries in fbcode.
        if not config.is_fbcode():
            libs += ["c10", "torch", "torch_cpu", "torch_python"]
            libs += ["gomp"]
        else:
            # internal remote execution is able to find omp, but not gomp
            libs += ["omp"]
        macros = vec_isa.build_macro()
        if macros:
            macros = f"-D{macros}"
        if cuda:
            if config.is_fbcode():
                libs += ["cuda"]
            else:
                libs += ["c10_cuda", "cuda", "torch_cuda"]
    else:
        # Note - this is effectively a header only inclusion. Usage of some header files may result in
        # symbol not found, if those header files require a library.
        # For those cases, include the lpath and libs command as we do for pytorch above.
        # This approach allows us to only pay for what we use.
        ipaths = cpp_extension.include_paths(cuda) + [sysconfig.get_path("include")]
        lpaths = []
        if sys.platform == "darwin":
            # GNU OpenMP generally is not available on MacOS
            # There is either Intel OpenMP(for x86) or LLVM OpenMP (for both x86 and arm64)
            libs = ["omp"]
            if os.getenv("CONDA_PREFIX") is not None:
                # On MacOS OpenMP is not available via the system install
                # But on conda can be provided using https://anaconda.org/anaconda/llvm-openmp
                conda_lib_path = os.path.join(os.getenv("CONDA_PREFIX"), "lib")
                ipaths.append(os.path.join(os.getenv("CONDA_PREFIX"), "include"))
                lpaths.append(conda_lib_path)
                # Prefer Intel OpenMP on x86 machine
                if os.uname().machine == "x86_64" and os.path.exists(
                    os.path.join(conda_lib_path, "libiomp5.dylib")
                ):
                    libs = ["iomp5"]
        else:
            libs = ["omp"] if config.is_fbcode() else ["gomp"]

    # third party libs
    if config.is_fbcode():
        ipaths.append(build_paths.sleef())
        ipaths.append(build_paths.openmp())
        # We also need to bundle includes with absolute paths into a remote directory
        # (later on, we copy the include paths from cpp_extensions into our remote dir)
        ipaths.append("include")

    ipaths = " ".join(["-I" + p for p in ipaths])
    lpaths = " ".join(["-L" + p for p in lpaths])
    libs = " ".join(["-l" + p for p in libs])
    return ipaths, lpaths, libs, macros


def cpp_compile_command(
    input,
    output,
    warning_all=True,
    shared=True,
    include_pytorch=False,
    vec_isa: VecISA = invalid_vec_isa,
    cuda=False,
    aot_mode=False,
):
    ipaths, lpaths, libs, macros = get_include_and_linking_paths(
        include_pytorch, vec_isa, cuda, aot_mode
    )
    if config.is_fbcode():
        # We need to copy any absolute-path torch includes
        inp_name = os.path.basename(input)
        out_name = os.path.basename(output)
        linker_path = f"-B{os.path.dirname(build_paths.ld())}"
    else:
        inp_name = input
        out_name = output
        linker_path = ""  # let the compiler pick
    return re.sub(
        r"[ \n]+",
        " ",
        f"""
            {cpp_compiler()} {inp_name} {get_shared(shared)}
            {get_warning_all_flag(warning_all)} {cpp_flags()}
            {ipaths} {lpaths} {libs} {macros} {linker_path}
            {optimization_flags()}
            {use_custom_generated_macros()}
            {use_fb_internal_macros()}
            -o {out_name}
        """,
    ).strip()


class CudaKernelParamCache:
    cache = dict()
    clear = staticmethod(cache.clear)

    @classmethod
    def set(cls, key, params, cubin):
        _, path = write(cubin, "cubin", hash_type="cubin")
        params["cubin_path"] = path
        cls.cache[key] = params

    @classmethod
    def get(cls, key):
        return cls.cache.get(key, None)


class AotCodeCache:
    cache = dict()
    clear = staticmethod(cache.clear)

    @classmethod
    def compile(cls, graph, source_code, cuda):
        # TODO: update cpp_compile_command for different platforms
        picked_vec_isa = invalid_vec_isa if cuda else pick_vec_isa()
        cpp_command = repr(
            cpp_compile_command(
                "i", "o", vec_isa=picked_vec_isa, cuda=cuda, aot_mode=graph.aot_mode
            )
        )
        key, input_path = write(source_code, "cpp", extra=cpp_command)
        if key not in cls.cache:
            from filelock import FileLock

            lock_dir = get_lock_dir()
            lock = FileLock(os.path.join(lock_dir, key + ".lock"), timeout=LOCK_TIMEOUT)
            with lock:
                output_so_dir = input_path[:-4]
                if not os.path.exists(output_so_dir):
                    os.makedirs(output_so_dir, exist_ok=False)
                so_name = f"{config.dll_name}.so"
                output_so = os.path.join(output_so_dir, so_name)
                if not os.path.exists(output_so):
                    cmd = cpp_compile_command(
                        input=input_path,
                        output=output_so,
                        vec_isa=picked_vec_isa,
                        cuda=cuda,
                        aot_mode=graph.aot_mode,
                    ).split(" ")
                    log.debug("aot compilation command: %s", " ".join(cmd))
                    try:
                        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
                    except subprocess.CalledProcessError as e:
                        raise exc.CppCompileError(cmd, e.output) from e

                cls.cache[key] = output_so

        def wrapper_call(*args):
            assert len(graph.graph_outputs) > 0
            return cls.cache[key], *(None for i in range(len(graph.graph_outputs) - 1))

        return wrapper_call


# Given a path to an input cpp file and an output path,
# Attempts to compile the file, storing the output in "output_path"
def compile_file(input_path, output_path, cmd) -> None:
    input_file = os.path.basename(input_path) if config.is_fbcode() else input_path
    try:
        if config.is_fbcode():
            # Need to copy our header into the same folder as the sourcecode.
            from torch._inductor.codegen.cpp import cpp_prefix_path

            header_path = cpp_prefix_path()
            header_name = os.path.basename(header_path)
            output_name = os.path.basename(output_path)
            # When we build remotely, we need to make sure to carefully copy any files
            # that are required during the compilation process into our build directly.
            # This is where all of the ATen/c10/Torch includes come from.
            torch_includes_path = os.path.join(
                torch.utils.cpp_extension._TORCH_PATH, "include"
            )
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Copy everything to tmp compilation folder
                shutil.copy(header_path, os.path.join(tmp_dir, header_name))
                shutil.copy(input_path, os.path.join(tmp_dir, input_file))
                dest_include_path = os.path.join(tmp_dir, "include")
                shutil.copytree(torch_includes_path, dest_include_path)
                # Run the build
                output_file_path = _run_build_command(cmd, tmp_dir, output_name)
                # Copy output from the build
                if os.path.exists(output_path):
                    os.remove(output_path)
                shutil.copy(output_file_path, output_path)
        else:
            subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise exc.CppCompileError(cmd, e.output) from e


class CppCodeCache:
    cache = dict()
    clear = staticmethod(cache.clear)

    @staticmethod
    def _load_library(path):
        try:
            return cdll.LoadLibrary(path)
        except OSError as e:
            if "gomp" in str(e) and os.path.exists("/usr/lib64/libgomp.so.1"):
                # hacky workaround for fbcode/buck
                global _libgomp
                _libgomp = cdll.LoadLibrary("/usr/lib64/libgomp.so.1")
                return cdll.LoadLibrary(path)
            if "failed to map segment from shared object" in str(e):
                raise OSError(
                    f"{e}.  The most common reason this may occur is if the {tempfile.gettempdir()} folder "
                    "is mounted with noexec (e.g., by default Docker mounts tmp file systems "
                    f"as noexec).  Please remount {tempfile.gettempdir()} with exec enabled, or set another "
                    "temporary directory with TORCHINDUCTOR_CACHE_DIR environment variable."
                ) from e
            raise

    @classmethod
    def load(cls, source_code):
        picked_vec_isa = pick_vec_isa()
        cpp_command = repr(cpp_compile_command("i", "o", vec_isa=picked_vec_isa))
        key, input_path = write(source_code, "cpp", extra=cpp_command)
        if key not in cls.cache:
            from filelock import FileLock

            lock_dir = get_lock_dir()
            lock = FileLock(os.path.join(lock_dir, key + ".lock"), timeout=LOCK_TIMEOUT)
            with lock:
                output_path = input_path[:-3] + "so"
                if not os.path.exists(output_path):
                    cmd = cpp_compile_command(
                        input=input_path, output=output_path, vec_isa=picked_vec_isa
                    ).split(" ")
                    compile_file(input_path, output_path, cmd)
                cls.cache[key] = cls._load_library(output_path)
                cls.cache[key].key = key

        return cls.cache[key]


class PyCodeCache:
    cache = dict()
    linemaps = dict()
    clear = staticmethod(cache.clear)

    @classmethod
    def write(cls, source_code, extra=""):
        return write(source_code, "py", extra=extra)

    @classmethod
    def load(cls, source_code, extra="", linemap=()):
        key, path = write(source_code, "py", extra=extra)
        return cls.load_by_key_path(key, path, linemap)

    @classmethod
    def load_by_key_path(cls, key, path, linemap=()):
        if key not in cls.cache:
            with open(path) as f:
                try:
                    code = compile(f.read(), path, "exec")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to import {path}\n{type(e).__name__}: {e}"
                    )
                mod = types.ModuleType(f"{__name__}.{key}")
                mod.__file__ = path
                mod.key = key
                exec(code, mod.__dict__, mod.__dict__)
                sys.modules[mod.__name__] = mod
                # another thread might set this first
                cls.cache.setdefault(key, mod)
                # unzip into separate lines/nodes lists
                cls.linemaps[path] = list(zip(*linemap))

        return cls.cache[key]

    @classmethod
    @functools.lru_cache(None)
    def stack_frames_for_code(cls, path, lineno):
        if path not in cls.linemaps:
            return None
        # [(starting_line, <fx node>), ...]
        lines, nodes = cls.linemaps[path]
        p = bisect_right(lines, lineno)
        if p == 0:
            return None
        entry = nodes[p - 1]
        if not entry:
            return None

        def parse_stack_trace(stack_trace):
            # ideally fx stores stack traces as data rather than a string
            # but this is not along a performance critical path
            regex = r'File "(.+)", line (\d+), in (.+)\n'
            matches = re.findall(regex, stack_trace)
            return [
                {"filename": f, "line": int(l), "name": n}
                for f, l, n in reversed(matches)
            ]

        return parse_stack_trace(entry)


class CppWrapperCodeCache:
    cache = dict()
    clear = staticmethod(cache.clear)

    @classmethod
    def load(cls, source_code, func_name, key, cuda):
        name = f"inline_extension_{key}"
        cpp_wrapper_dir = cpp_wrapper_cache_dir(name)
        if not os.path.exists(cpp_wrapper_dir):
            os.makedirs(cpp_wrapper_dir)

        ext = "so"
        filepath = os.path.join(cpp_wrapper_dir, f"{name}.{ext}")
        log.debug("Cpp wrapper code path %s", filepath)

        if key not in cls.cache:
            log.debug("Cpp wrapper cache miss for %s", filepath)
            from filelock import FileLock

            lock_dir = get_lock_dir()
            lock = FileLock(os.path.join(lock_dir, key + ".lock"), timeout=LOCK_TIMEOUT)
            with lock:
                if not os.path.exists(filepath):
                    log.debug("Cpp wrapper building %s", filepath)

                    _cpp_flags = cpp_flags()
                    _opt_flags = optimization_flags()
                    _shared = get_shared()
                    _warning_all_flag = get_warning_all_flag()
                    _ipaths, _lpaths, _libs, _macros = get_include_and_linking_paths(
                        vec_isa=pick_vec_isa(),
                        cuda=cuda,
                    )
                    _use_custom_generated_macros = use_custom_generated_macros()

                    extra_cflags = f"{_cpp_flags} {_opt_flags} {_warning_all_flag} {_macros} {_use_custom_generated_macros}"
                    # For CPP wrapper, add -ffast-math during linking to make CPU flush denormals.
                    # CPP wrapper leverages cpp_extension which will do the compilation and linking in two stages.
                    # We need to explicitly add -ffast-math as a linking flag.
                    # For the default python wrapper, the compilation and linking are done in one command thus -ffast-math
                    # will take effect in both compilation and linking.
                    extra_ldflags = f"{_shared} {_lpaths} {_libs} -ffast-math"
                    extra_include_paths = f"{_ipaths}"

                    mod = torch.utils.cpp_extension.load_inline(
                        name=name,
                        build_directory=cpp_wrapper_dir,
                        cpp_sources=[source_code],
                        functions=[func_name],
                        extra_cflags=[extra_cflags],
                        extra_ldflags=[extra_ldflags],
                        extra_include_paths=[extra_include_paths],
                    )
                    log.debug("Cpp wrapper done building %s", filepath)
                else:
                    log.debug("Found target .so, cpp wrapper loading %s", filepath)
                    spec = importlib.util.spec_from_file_location(name, filepath)
                    assert spec is not None
                    mod = importlib.util.module_from_spec(spec)
                    assert isinstance(spec.loader, abc.Loader)
                    spec.loader.exec_module(mod)
                    log.debug("Cpp wrapper done loading %s", filepath)

                cls.cache[key] = mod

        return cls.cache[key]


class TritonCodeCache:
    @classmethod
    def load(cls, kernel_name, source_code):
        mod = PyCodeCache.load(source_code)
        return getattr(mod, kernel_name)


def _worker_compile(kernel_name, source_code, cc, device):
    cuda_properties.set_compiler_worker_current_device(device)
    kernel = TritonCodeCache.load(kernel_name, source_code)
    kernel.precompile(warm_cache_only_with_cc=cc)


def _load_kernel(kernel_name, source_code):
    kernel = TritonCodeCache.load(kernel_name, source_code)
    kernel.precompile()
    return kernel


class TritonFuture:
    def __init__(self, kernel_name, source_code, future):
        self.kernel_name = kernel_name
        self.source_code = source_code
        self.future = future

    # @dynamo_utils.dynamo_timed
    def result(self):
        t0 = time()
        if hasattr(self, "kernel"):
            return self.kernel
        # If the worker failed this will throw an exception.
        self.future.result()
        kernel = self.kernel = _load_kernel(self.kernel_name, self.source_code)
        latency = time() - t0
        if latency > 50:
            developer_warning(
                f"Detected long compilation time of {latency} seconds for kernel name {self.kernel_name}"
            )
            developer_warning(self.source_code)
        del self.kernel_name, self.source_code, self.future
        return kernel


class AsyncCompile:
    def __init__(self):
        pass

    @staticmethod
    @functools.lru_cache(1)
    def pool():
        assert config.compile_threads > 1
        return ThreadPoolExecutor(config.compile_threads)

    @staticmethod
    @functools.lru_cache(1)
    def process_pool():
        # ensure properties have been calculated before processes
        # are forked
        cuda_properties._properties()
        assert config.compile_threads > 1
        orig_ppid = os.getpid()

        # if this process dies abnormally (e.g. segfault)
        # it will not shut down the workers. Instead
        # the workers will have their parent reassigned to the
        # init process. This launches a separate thread to
        # watch for the worker getting reassigned,
        # and cleans it up in this case.
        def init():
            def run():
                while True:
                    sleep(1)
                    if orig_ppid != os.getppid():
                        os.kill(os.getpid(), signal.SIGKILL)

            global _watchdog_thread
            _watchdog_thread = Thread(target=run, daemon=True)
            _watchdog_thread.start()

        # we rely on 'fork' because we cannot control whether users
        # have an `if __name__ == '__main__'` in their main process.
        fork_context = multiprocessing.get_context("fork")
        pool = ProcessPoolExecutor(
            config.compile_threads, mp_context=fork_context, initializer=init
        )
        # when this pool is created in a subprocess object, the normal exit handler
        # doesn't run, and we need to register our own handler.
        # exitpriority has to be high, because another one of the finalizers will
        # kill the worker thread that sends the shutdown message to the workers...
        multiprocessing.util.Finalize(None, pool.shutdown, exitpriority=sys.maxsize)
        return pool

    @classmethod
    def warm_pool(cls):
        if config.compile_threads <= 1:
            return
        _compile_start()
        pool = cls.process_pool()

        # We have to fork processes for compiler workers, but the more memory and other resources that are loaded, the
        # slower the os.fork time is, quite drastically. It also holds the GIL so we can't put it on another thread.

        # Examples:
        # A simple x + x + x script: 10ms seconds in the middle of the program, 2ms at startup
        # tf_efficientnet_b0 benchmark: 50ms! in the middle of the program , 3ms at startup

        # So we want to start the workers early when it is still cheap, and also to allow the workers to get
        # ready before we have work for them.

        # ProcessPoolExecutor also does not launch the workers until it finds a point when all the workers are idle.
        # But if we waited until then fork time will be long and we will be waiting for the processes to initialize.

        # We force them to start here with some YOLOing of the internal methods.
        if hasattr(pool, "_start_queue_management_thread"):
            pool._start_queue_management_thread()
        else:
            for _ in range(config.compile_threads):
                pool._adjust_process_count()
            pool._start_executor_manager_thread()
        _compile_end()

    @classmethod
    def submit(cls, task):
        if config.compile_threads <= 1:
            return task()
        return cls.pool().submit(task)

    @classmethod
    def map(cls, fn, seq):
        if config.compile_threads <= 1 or len(seq) <= 1:
            return list(map(fn, seq))
        return [t.result() for t in [cls.pool().submit(fn, x) for x in seq]]

    def triton(self, kernel_name, source_code):
        _compile_start()

        if config.compile_threads > 1:
            major, minor = torch.cuda.get_device_capability()
            device = torch.cuda.current_device()
            cc = major * 10 + minor
            future = self.process_pool().submit(
                _worker_compile, kernel_name, source_code, cc, device
            )
            return TritonFuture(kernel_name, source_code, future)
        else:
            return _load_kernel(kernel_name, source_code)

    def cpp(self, source_code):
        def task():
            return CppCodeCache.load(source_code).kernel

        return self.submit(task)

    def wait(self, scope: Dict[str, Any]):
        num_kernels = len(
            [
                value
                for key, value in scope.items()
                if isinstance(value, (Future, TritonFuture))
            ]
        )
        pbar = tqdm(
            total=num_kernels,
            desc="Inductor Compilation",
            disable=config.disable_progress,
            delay=0,
        )
        if config.compile_threads > 1:
            for key, result in scope.items():
                if config.verbose_progress and not isinstance(pbar, _Faketqdm):
                    pbar.set_postfix_str(key)
                if isinstance(result, (Future, TritonFuture)):
                    scope[key] = result.result()
                    pbar.update(1)

        _compile_end()


AsyncCompile.warm_pool()
