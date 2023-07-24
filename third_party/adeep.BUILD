#*******************************************************************************
# Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
#*******************************************************************************

load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "adeep",
    hdrs = glob([
        "include/**/*.hpp",
        "include/**/*.h",
    ]),
    defines = [
        "ADEEP_USE_BLIS",
    ],
    includes = [
        "include/",
    ],
    visibility = ["//visibility:public"],
    deps = ["@zendnn//:zendnn"],
)
