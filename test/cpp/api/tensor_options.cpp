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

#include <gtest/gtest.h>
#include <test/cpp/api/support.h>

#include <torch/torch.h>

#include <string>
#include <vector>

using namespace at;
using namespace torch::test;

// A macro so we don't lose location information when an assertion fails.
#define REQUIRE_OPTIONS(device_, index_, type_, layout_)                  \
  ASSERT_EQ(options.device().type(), Device((device_), (index_)).type()); \
  ASSERT_TRUE(                                                            \
      options.device().index() == Device((device_), (index_)).index());   \
  ASSERT_EQ(options.dtype(), (type_));                                    \
  ASSERT_TRUE(options.layout() == (layout_))

#define REQUIRE_TENSOR_OPTIONS(device_, index_, type_, layout_)            \
  ASSERT_EQ(tensor.device().type(), Device((device_), (index_)).type());   \
  ASSERT_EQ(tensor.device().index(), Device((device_), (index_)).index()); \
  ASSERT_EQ(tensor.scalar_type(), (type_));                                \
  ASSERT_TRUE(tensor.options().layout() == (layout_))

TEST(TensorOptionsTest, DefaultsToTheRightValues) {
  TensorOptions options;
  REQUIRE_OPTIONS(kCPU, -1, kFloat, kStrided);
}

TEST(TensorOptionsTest, UtilityFunctionsReturnTheRightTensorOptions) {
  auto options = dtype(kInt);
  REQUIRE_OPTIONS(kCPU, -1, kInt, kStrided);

  options = layout(kSparse);
  REQUIRE_OPTIONS(kCPU, -1, kFloat, kSparse);

  options = device({kCUDA, 1});
  REQUIRE_OPTIONS(kCUDA, 1, kFloat, kStrided);

  options = device_index(1);
  REQUIRE_OPTIONS(kCUDA, 1, kFloat, kStrided);

  options = dtype(kByte).layout(kSparse).device(kCUDA, 2).device_index(3);
  REQUIRE_OPTIONS(kCUDA, 3, kByte, kSparse);
}

TEST(TensorOptionsTest, ConstructsWellFromCPUTypes) {
  TensorOptions options;
  REQUIRE_OPTIONS(kCPU, -1, kFloat, kStrided);

  options = TensorOptions({kCPU, 0});
  REQUIRE_OPTIONS(kCPU, 0, kFloat, kStrided);

  options = TensorOptions("cpu:0");
  REQUIRE_OPTIONS(kCPU, 0, kFloat, kStrided);

  options = TensorOptions(kInt);
  REQUIRE_OPTIONS(kCPU, -1, kInt, kStrided);

  options =
      TensorOptions(getDeprecatedTypeProperties(Backend::SparseCPU, kFloat));
  REQUIRE_OPTIONS(kCPU, -1, kFloat, kSparse);

  options =
      TensorOptions(getDeprecatedTypeProperties(Backend::SparseCPU, kByte));
  REQUIRE_OPTIONS(kCPU, -1, kByte, kSparse);
}

TEST(TensorOptionsTest, ConstructsWellFromCPUTensors) {
  auto options = empty(5, kDouble).options();
  REQUIRE_OPTIONS(kCPU, -1, kDouble, kStrided);

  options = empty(5, getDeprecatedTypeProperties(Backend::SparseCPU, kByte))
                .options();
  REQUIRE_OPTIONS(kCPU, -1, kByte, kSparse);
}

TEST(TensorOptionsTest, ConstructsWellFromVariables) {
  auto options = torch::empty(5).options();
  REQUIRE_OPTIONS(kCPU, -1, kFloat, kStrided);
  ASSERT_FALSE(options.requires_grad());

  options = torch::empty(5, at::requires_grad()).options();
  REQUIRE_OPTIONS(kCPU, -1, kFloat, kStrided);
  ASSERT_FALSE(options.requires_grad());
}

TEST(DeviceTest, ParsesCorrectlyFromString) {
  Device device("cpu:0");
  ASSERT_EQ(device, Device(DeviceType::CPU, 0));

  device = Device("cpu");
  ASSERT_EQ(device, Device(DeviceType::CPU));

  device = Device("cuda:123");
  ASSERT_EQ(device, Device(DeviceType::CUDA, 123));

  device = Device("cuda");
  ASSERT_EQ(device, Device(DeviceType::CUDA));

  device = Device("mkldnn");
  ASSERT_EQ(device, Device(DeviceType::MKLDNN));

  device = Device("zendnn");
  ASSERT_EQ(device, Device(DeviceType::ZENDNN));

  device = Device("opengl");
  ASSERT_EQ(device, Device(DeviceType::OPENGL));

  device = Device("opencl");
  ASSERT_EQ(device, Device(DeviceType::OPENCL));

  device = Device("ideep");
  ASSERT_EQ(device, Device(DeviceType::IDEEP));

  device = Device("hip");
  ASSERT_EQ(device, Device(DeviceType::HIP));

  device = Device("hip:123");
  ASSERT_EQ(device, Device(DeviceType::HIP, 123));

  std::vector<std::string> badnesses = {
      "", "cud:1", "cuda:", "cpu::1", ":1", "3", "tpu:4", "??"};
  for (const auto& badness : badnesses) {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
    ASSERT_ANY_THROW({ Device d(badness); });
  }
}

TEST(DefaultDtypeTest, CanSetAndGetDefaultDtype) {
  AutoDefaultDtypeMode dtype_mode(kFloat);

  ASSERT_EQ(at::get_default_dtype(), kFloat);
  set_default_dtype(caffe2::TypeMeta::Make<int>());
  ASSERT_EQ(at::get_default_dtype(), kInt);
}

TEST(DefaultDtypeTest, NewTensorOptionsHasCorrectDefault) {
  AutoDefaultDtypeMode dtype_mode(kFloat);

  set_default_dtype(caffe2::TypeMeta::Make<int>());
  ASSERT_EQ(at::get_default_dtype(), kInt);
  TensorOptions options;
  ASSERT_EQ(options.dtype(), kInt);
}

TEST(DefaultDtypeTest, NewTensorsHaveCorrectDefaultDtype) {
  AutoDefaultDtypeMode dtype_mode(kFloat);
  set_default_dtype(caffe2::TypeMeta::Make<int>());
  {
    auto tensor = torch::ones(5);
    ASSERT_EQ(tensor.dtype(), kInt);
  }
  set_default_dtype(caffe2::TypeMeta::Make<double>());
  {
    auto tensor = torch::ones(5);
    ASSERT_EQ(tensor.dtype(), kDouble);
  }
  {
    auto tensor = torch::ones(5, kFloat);
    ASSERT_EQ(tensor.dtype(), kFloat);
  }
}
