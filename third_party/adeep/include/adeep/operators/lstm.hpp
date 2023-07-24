/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#ifndef ADEEP_OPERATORS_LSTM_HPP
#define ADEEP_OPERATORS_LSTM_HPP

namespace adeep {

struct lstm_forward : public zendnn::lstm_forward {
  static void compute() {
  }
};

struct lstm_backward : public zendnn::lstm_backward {
  static void compute() {
  }
};

}  // namespace adeep

#endif