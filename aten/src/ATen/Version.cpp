/******************************************************************************
* Modifications Copyright (c) 2023 Advanced Micro Devices, Inc.
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice,
* this list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
* 3. Neither the name of the copyright holder nor the names of its contributors
* may be used to endorse or promote products derived from this software without
* specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
* LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
*
******************************************************************************/

#include <ATen/Version.h>
#include <ATen/Config.h>

#if AT_MKL_ENABLED()
#include <mkl.h>
#endif

#if AT_ZENDNN_ENABLED()
#include <blis/blis.h>
#include <zendnn_version.h>
#endif

#if AT_MKLDNN_ENABLED()
#include <dnnl.hpp>
#include <ideep.hpp>
#endif

#include <caffe2/core/common.h>

#include <ATen/native/DispatchStub.h>

#include <sstream>

namespace at {

std::string get_mkl_version() {
  std::string version;
  #if AT_MKL_ENABLED()
    {
      // Magic buffer number is from MKL documentation
      // https://software.intel.com/en-us/mkl-developer-reference-c-mkl-get-version-string
      char buf[198];
      mkl_get_version_string(buf, 198);
      version = buf;
    }
  #else
    version = "MKL not found";
  #endif
  return version;
}

std::string get_mkldnn_version() {
  std::ostringstream ss;
  #if AT_MKLDNN_ENABLED()
    // Cribbed from mkl-dnn/src/common/verbose.cpp
    // Too bad: can't get ISA info conveniently :(
    // Apparently no way to get ideep version?
    // https://github.com/intel/ideep/issues/29
    {
      const dnnl_version_t* ver = dnnl_version();
      ss << "Intel(R) MKL-DNN v" << ver->major << "." << ver->minor << "." << ver->patch
         << " (Git Hash " << ver->hash << ")";
    }
  #else
    ss << "MKLDNN not found";
  #endif
  return ss.str();
}

#if AT_ZENDNN_ENABLED()

    std::string get_zendnn_version() {
    std::ostringstream ss;
    ss << ZENDNN_VERSION_MAJOR << "." << ZENDNN_VERSION_MINOR << "." << ZENDNN_VERSION_PATCH;
    return ss.str();
    }

#endif

std::string get_openmp_version() {
  std::ostringstream ss;
  #ifdef _OPENMP
    {
      ss << "OpenMP " << _OPENMP;
      const char* ver_str = nullptr;
      switch (_OPENMP) {
        case 200505:
          ver_str = "2.5";
          break;
        case 200805:
          ver_str = "3.0";
          break;
        case 201107:
          ver_str = "3.1";
          break;
        case 201307:
          ver_str = "4.0";
          break;
        case 201511:
          ver_str = "4.5";
          break;
        default:
          ver_str = nullptr;
          break;
      }
      if (ver_str) {
        ss << " (a.k.a. OpenMP " << ver_str << ")";
      }
    }
  #else
    ss << "OpenMP not found";
  #endif
  return ss.str();
}

std::string used_cpu_capability() {
  // It is possible that we override the cpu_capability with
  // environment variable
  std::ostringstream ss;
  ss << "CPU capability usage: ";
  auto capability = native::get_cpu_capability();
  switch (capability) {
#if defined(HAVE_VSX_CPU_DEFINITION)
    case native::CPUCapability::DEFAULT:
      ss << "DEFAULT";
      break;
    case native::CPUCapability::VSX:
      ss << "VSX";
      break;
#elif defined(HAVE_ZVECTOR_CPU_DEFINITION)
    case native::CPUCapability::DEFAULT:
      ss << "DEFAULT";
      break;
    case native::CPUCapability::ZVECTOR:
      ss << "Z VECTOR";
      break;
#else
    case native::CPUCapability::DEFAULT:
      ss << "NO AVX";
      break;
    case native::CPUCapability::AVX2:
      ss << "AVX2";
      break;
    case native::CPUCapability::AVX512:
      ss << "AVX512";
      break;
#endif
    default:
      break;
  }
  return ss.str();
}

std::string show_config() {
  std::ostringstream ss;
  ss << "PyTorch built with:\n";

  // Reference:
  // https://blog.kowalczyk.info/article/j/guide-to-predefined-macros-in-c-compilers-gcc-clang-msvc-etc..html

#if defined(__GNUC__)
  {
    ss << "  - GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "\n";
  }
#endif

#if defined(__cplusplus)
  {
    ss << "  - C++ Version: " << __cplusplus << "\n";
  }
#endif

#if defined(__clang_major__)
  {
    ss << "  - clang " << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__ << "\n";
  }
#endif

#if defined(_MSC_VER)
  {
    ss << "  - MSVC " << _MSC_FULL_VER << "\n";
  }
#endif

#if AT_MKL_ENABLED()
  ss << "  - " << get_mkl_version() << "\n";
#endif

#if AT_MKLDNN_ENABLED()
  ss << "  - " << get_mkldnn_version() << "\n";
#endif

#if AT_ZENDNN_ENABLED()
  ss << "  - " <<  "AMD " << bli_info_get_version_str() << " ( Git Hash " << BLIS_VERSION_HASH << " )" << "\n";
  ss << "  - " <<  "AMD ZENDNN v" << get_zendnn_version() << " ( Git Hash " << ZENDNN_PT_VERSION_HASH << " )" << "\n";
#endif

#ifdef _OPENMP
  ss << "  - " << get_openmp_version() << "\n";
#endif

#if AT_BUILD_WITH_LAPACK()
  // TODO: Actually record which one we actually picked
  ss << "  - LAPACK is enabled (usually provided by MKL)\n";
#endif

#if AT_NNPACK_ENABLED()
  // TODO: No version; c.f. https://github.com/Maratyszcza/NNPACK/issues/165
  ss << "  - NNPACK is enabled\n";
#endif

#ifdef CROSS_COMPILING_MACOSX
  ss << "  - Cross compiling on MacOSX\n";
#endif

  ss << "  - "<< used_cpu_capability() << "\n";

  if (hasCUDA()) {
    ss << detail::getCUDAHooks().showConfig();
  }

  if (hasORT()) {
    ss << detail::getORTHooks().showConfig();
  }

  ss << "  - Build settings: ";
  for (const auto& pair : caffe2::GetBuildOptions()) {
    if (!pair.second.empty()) {
      ss << pair.first << "=" << pair.second << ", ";
    }
  }
  ss << "\n";

  // TODO: do HIP
  // TODO: do XLA
  // TODO: do MPS

  return ss.str();
}

std::string get_cxx_flags() {
  #if defined(FBCODE_CAFFE2)
  TORCH_CHECK(
    false,
    "Buck does not populate the `CXX_FLAGS` field of Caffe2 build options. "
    "As a result, `get_cxx_flags` is OSS only."
  );
  #endif
  return caffe2::GetBuildOptions().at("CXX_FLAGS");
}

}
