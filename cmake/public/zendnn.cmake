#*******************************************************************************
# Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
#*******************************************************************************

set(ZENDNN_USE_NATIVE_ARCH ${USE_NATIVE_ARCH})

find_package(ZENDNN QUIET)

if(NOT TARGET caffe2::zendnn)
  add_library(caffe2::zendnn INTERFACE IMPORTED)
endif()

set_property(
  TARGET caffe2::zendnn PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  ${ZENDNN_INCLUDE_DIR})
set_property(
  TARGET caffe2::zendnn PROPERTY INTERFACE_LINK_LIBRARIES
  ${ZENDNN_LIBRARIES})
