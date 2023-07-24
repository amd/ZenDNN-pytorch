/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#ifndef ADEEP_OPERATORS_LBR_GRU_HPP
#define ADEEP_OPERATORS_LBR_GRU_HPP

namespace adeep {

struct lbr_gru_forward : public zendnn::lbr_gru_forward {
  static void compute() {
  }
};

struct lbr_gru_backward : public zendnn::lbr_gru_backward {
  static void compute() {
  }
};

}  // namespace adeep

#endif