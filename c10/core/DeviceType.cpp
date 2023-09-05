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

#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>

namespace c10 {

std::string DeviceTypeName(DeviceType d, bool lower_case) {
  switch (d) {
    // I considered instead using ctype::tolower to lower-case the strings
    // on the fly, but this seemed a bit much.
    case DeviceType::CPU:
      return lower_case ? "cpu" : "CPU";
    case DeviceType::CUDA:
      return lower_case ? "cuda" : "CUDA";
    case DeviceType::OPENGL:
      return lower_case ? "opengl" : "OPENGL";
    case DeviceType::OPENCL:
      return lower_case ? "opencl" : "OPENCL";
    case DeviceType::MKLDNN:
      return lower_case ? "mkldnn" : "MKLDNN";
    case DeviceType::ZENDNN:
      return lower_case ? "zendnn" : "ZENDNN";
    case DeviceType::IDEEP:
      return lower_case ? "ideep" : "IDEEP";
    case DeviceType::HIP:
      return lower_case ? "hip" : "HIP";
    case DeviceType::VE:
      return lower_case ? "ve" : "VE";
    case DeviceType::FPGA:
      return lower_case ? "fpga" : "FPGA";
    case DeviceType::ORT:
      return lower_case ? "ort" : "ORT";
    case DeviceType::XLA:
      return lower_case ? "xla" : "XLA";
    case DeviceType::Lazy:
      return lower_case ? "lazy" : "LAZY";
    case DeviceType::MPS:
      return lower_case ? "mps" : "MPS";
    case DeviceType::Vulkan:
      return lower_case ? "vulkan" : "VULKAN";
    case DeviceType::Metal:
      return lower_case ? "metal" : "METAL";
    case DeviceType::XPU:
      return lower_case ? "xpu" : "XPU";
    case DeviceType::Meta:
      return lower_case ? "meta" : "META";
    case DeviceType::HPU:
      return lower_case ? "hpu" : "HPU";
    case DeviceType::IPU:
      return lower_case ? "ipu" : "IPU";
    case DeviceType::PrivateUse1:
      return lower_case ? "privateuseone" : "PRIVATEUSEONE";
    default:
      TORCH_CHECK(
          false,
          "Unknown device: ",
          static_cast<int16_t>(d),
          ". If you have recently updated the caffe2.proto file to add a new "
          "device type, did you forget to update the DeviceTypeName() "
          "function to reflect such recent changes?");
      // The below code won't run but is needed to suppress some compiler
      // warnings.
      return "";
  }
}

// NB: Per the C++ standard
// as long as you cast from the same underlying type, it is always valid to cast
// into an enum class (even if the value would be invalid by the enum.)  Thus,
// the caller is allowed to cast a possibly invalid int16_t to DeviceType and
// then pass it to this function.  (I considered making this function take an
// int16_t directly, but that just seemed weird.)
bool isValidDeviceType(DeviceType d) {
  switch (d) {
    case DeviceType::CPU:
    case DeviceType::CUDA:
    case DeviceType::OPENGL:
    case DeviceType::OPENCL:
    case DeviceType::MKLDNN:
    case DeviceType::ZENDNN:
    case DeviceType::IDEEP:
    case DeviceType::HIP:
    case DeviceType::VE:
    case DeviceType::FPGA:
    case DeviceType::ORT:
    case DeviceType::XLA:
    case DeviceType::Lazy:
    case DeviceType::MPS:
    case DeviceType::Vulkan:
    case DeviceType::Metal:
    case DeviceType::XPU:
    case DeviceType::Meta:
    case DeviceType::HPU:
    case DeviceType::IPU:
    case DeviceType::PrivateUse1:
      return true;
    default:
      return false;
  }
}

std::ostream& operator<<(std::ostream& stream, DeviceType type) {
  stream << DeviceTypeName(type, /* lower case */ true);
  return stream;
}

} // namespace c10
