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

#ifndef ADEEP_OPERATORS_SPLITER_HPP
#define ADEEP_OPERATORS_SPLITER_HPP

namespace adeep {

struct spliter {

  static std::vector<tensor> compute(const tensor& input,
                                     std::vector<int32_t>& axis_info,
                                     int axis,
                                     bool add_axis = false) {
    std::vector<tensor> outputs;
    tensor::dims output_dims(input.get_dims());
    tensor::dims offset_dims(output_dims.size(), 0);
    ADEEP_ENFORCE(axis < input.ndims(), "invalid axis in split");

    for (auto i = 0; i < axis_info.size(); ++i) {
      output_dims[axis] = axis_info[i];
      auto output = input.extract_submemory(output_dims, offset_dims);

      if (input.has_scale()) {
        output.set_scale(input.get_scale());
      }

      if (add_axis) {
        tensor::dims out_dims(output_dims);
        out_dims.erase(out_dims.begin() + axis);
        output.reshape(out_dims);
      }

      outputs.emplace_back(output);
      offset_dims[axis] += axis_info[i];
    }

    return outputs;
  }
};


}  // namespace adeep

#endif