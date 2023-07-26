/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#include <ATen/ATen.h>

#ifdef AT_CUDNN_ENABLED
#error "AT_CUDNN_ENABLED should not be visible in public headers"
#endif

#ifdef AT_MKL_ENABLED
#error "AT_MKL_ENABLED should not be visible in public headers"
#endif

#ifdef AT_MKLDNN_ENABLED
#error "AT_MKLDNN_ENABLED should not be visible in public headers"
#endif

#ifdef AT_ZENDNN_ENABLED
#error "AT_ZENDNN_ENABLED should not be visible in public headers"
#endif

#ifdef CAFFE2_STATIC_LINK_CUDA
#error "CAFFE2_STATIC_LINK_CUDA should not be visible in public headers"
#endif

auto main() -> int {}
