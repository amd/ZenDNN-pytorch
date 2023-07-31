#******************************************************************************
# Modifications Copyright (c) 2023 Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
#******************************************************************************

load("@rules_cc//cc:defs.bzl", "cc_library")
load("@//third_party:substitution.bzl", "template_rule")

template_rule(
    name = "include_zendnn_version",
    src = "include/zendnn_version.h.in",
    out = "include/zendnn_version.h",
    substitutions = {
        "@ZENDNN_VERSION_MAJOR@": "4",
        "@ZENDNN_VERSION_MINOR@": "1",
        "@ZENDNN_VERSION_PATCH@": "0",
    },
)

template_rule(
    name = "include_zendnn_config",
    src = "include/zendnn_config.h.in",
    out = "include/zendnn_config.h",
    substitutions = {
        "cmakedefine": "define",
        "${ZENDNN_CPU_THREADING_RUNTIME}": "OMP",
        "${ZENDNN_GPU_RUNTIME}": "NONE",
    },
)

cc_library(
    name = "zendnn",
    srcs = glob([
        "src/common/*.cpp",
        "src/cpu/**/*.cpp",
    ]),
    hdrs = glob([
        "include/*.h",
        "include/*.hpp",
        "src/*.hpp",
        "src/cpu/**/*.hpp",
        "src/cpu/**/*.h",
        "src/common/*.hpp",
        "src/cpu/rnn/*.hpp",
    ]) + [
        "include/zendnn_version.h",
        "include/zendnn_config.h",
    ],
    copts = [
        "-DUSE_AVX",
        "-DUSE_AVX2",
        "-DZENDNN_DLL",
        "-DZENDNN_DLL_EXPORTS",
        "-DZENDNN_ENABLE_CONCURRENT_EXEC",
        "-D__STDC_CONSTANT_MACROS",
        "-D__STDC_LIMIT_MACROS",
        "-fno-strict-overflow",
        "-fopenmp",
    ],
    includes = [
        "include/",
        "src/",
        "src/common/",
        "src/cpu/",
        "src/cpu/x64/xbyak/",
    ],
    visibility = ["//visibility:public"],
    linkopts = [
        "-lgomp",
    ],
    deps = [
        "@zendnn",
    ] + select({
        "@//tools/config:thread_sanitizer": [],
        "//conditions:default": ["@tbb"],
    }),
)
