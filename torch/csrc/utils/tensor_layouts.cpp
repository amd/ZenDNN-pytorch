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

#include <ATen/Layout.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/tensor_layouts.h>

namespace torch {
namespace utils {

#define REGISTER_LAYOUT(layout, LAYOUT)                                     \
  PyObject* layout##_layout =                                               \
      THPLayout_New(at::Layout::LAYOUT, "torch." #layout);                  \
  Py_INCREF(layout##_layout);                                               \
  if (PyModule_AddObject(torch_module, "" #layout, layout##_layout) != 0) { \
    throw python_error();                                                   \
  }                                                                         \
  registerLayoutObject((THPLayout*)layout##_layout, at::Layout::LAYOUT);

void initializeLayouts() {
  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch_module)
    throw python_error();

  PyObject* strided_layout =
      THPLayout_New(at::Layout::Strided, "torch.strided");
  Py_INCREF(strided_layout);
  if (PyModule_AddObject(torch_module, "strided", strided_layout) != 0) {
    throw python_error();
  }
  registerLayoutObject((THPLayout*)strided_layout, at::Layout::Strided);

  PyObject* sparse_coo_layout =
      THPLayout_New(at::Layout::Sparse, "torch.sparse_coo");
  Py_INCREF(sparse_coo_layout);
  if (PyModule_AddObject(torch_module, "sparse_coo", sparse_coo_layout) != 0) {
    throw python_error();
  }
  registerLayoutObject((THPLayout*)sparse_coo_layout, at::Layout::Sparse);

  REGISTER_LAYOUT(sparse_csr, SparseCsr)
  REGISTER_LAYOUT(sparse_csc, SparseCsc)
  REGISTER_LAYOUT(sparse_bsr, SparseBsr)
  REGISTER_LAYOUT(sparse_bsc, SparseBsc)

  PyObject* mkldnn_layout = THPLayout_New(at::Layout::Mkldnn, "torch._mkldnn");
  Py_INCREF(mkldnn_layout);
  if (PyModule_AddObject(torch_module, "_mkldnn", mkldnn_layout) != 0) {
    throw python_error();
  }
  registerLayoutObject((THPLayout*)mkldnn_layout, at::Layout::Mkldnn);

  PyObject *zendnn_layout = THPLayout_New(at::Layout::Zendnn, "torch._zendnn");
  Py_INCREF(zendnn_layout);
  if (PyModule_AddObject(torch_module, "_zendnn", zendnn_layout) != 0) {
    throw python_error();
  }
  registerLayoutObject((THPLayout*)zendnn_layout, at::Layout::Zendnn);

}

} // namespace utils
} // namespace torch
