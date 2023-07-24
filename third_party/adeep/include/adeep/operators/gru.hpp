/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#ifndef ADEEP_OPERATORS_GRU_HPP
#define ADEEP_OPERATORS_GRU_HPP

namespace adeep {

struct gru_forward : public zendnn::gru_forward {
  static void compute() {
  }
};

struct gru_backward : public zendnn::gru_backward {
  static void compute() {
  }
};

}  // namespace adeep

#endif