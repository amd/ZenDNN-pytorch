/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/
#include <map>
#include <tuple>
#include<iostream>
#include <ATen/Config.h>
#include <ATen/Utils.h>
#include <ATen/core/interned_strings.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/fold_conv_bn.h>
#include <torch/csrc/jit/passes/frozen_conv_folding.h>
#include <torch/csrc/jit/passes/frozen_ops_to_zendnn.h>
#include <torch/csrc/jit/passes/frozen_zendnn_graph_optimizations.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator_options.h>
#include <torch/csrc/jit/tensorexpr/types.h>
// clang-format off
// moving ConvUtils include induces import cycle
#include <ATen/native/ConvUtils.h>
#include <algorithm>
#include <memory>
#include <ATen/core/stack.h>
#include <c10/core/Layout.h>
#include <c10/util/StringUtil.h>

#if AT_ZENDNN_ENABLED()
#include <ATen/CPUFunctions.h>
#include <zendnn_types.h>
#include <ATen/native/zendnn/ZENDNNCommon.h>
#include <adeep.hpp>
#include "zendnn_helper.hpp"
#endif


namespace torch {
namespace jit {

#if AT_ZENDNN_ENABLED()
using Tensor = at::Tensor;
namespace {

jit::RegisterOperators reg_fut_ops_linear({
    jit::Operator(
        // XXX: this follows the schema convention of conv2d/conv3d, not
        // aten::zendnn_convolution, which is different for some reason!
        "prim::zendnn_linear_gelu(Tensor input, Tensor weight, Tensor? bias) -> Tensor",
        [](jit::Stack* stack) {
          Tensor bias;
          IValue bias_ival = pop(stack);
          if (!bias_ival.isNone()) {
            bias = bias_ival.toTensor();
          }
          Tensor weight = pop(stack).toTensor();
          Tensor input = pop(stack).toTensor();

          at::AutoDispatchBelowAutograd mode;

          TORCH_CHECK(
              input.options().type_equal(weight.options()),
              "Input type (",
              input.toString(),
              ") and weight type (",
              weight.toString(),
              ") should be the same");

          push(
              stack,
              at::native::zendnn_linear(
                  input, weight, bias, true));
        },
         AliasAnalysisKind::FROM_SCHEMA),
});



jit::RegisterOperators reg_fut_ops({
    jit::Operator(
        // XXX: this follows the schema convention of conv2d/conv3d, not
        // aten::zendnn_convolution, which is different for some reason!
        "prim::zendnn_convolution_relu(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor",
        [](jit::Stack* stack) {
          int64_t groups = pop(stack).toInt();
          auto dilation = pop(stack).toIntVector();
          auto padding = pop(stack).toIntVector();
          auto stride = pop(stack).toIntVector();

          Tensor bias;
          IValue bias_ival = pop(stack);
          if (!bias_ival.isNone()) {
            bias = bias_ival.toTensor();
          }
          Tensor weight = pop(stack).toTensor();
          Tensor input = pop(stack).toTensor();

          at::AutoDispatchBelowAutograd mode;
          // aten::convolution takes care of 0 dim case before calls into
          // backends
          if (input.size(0) == 0) {
            std::vector<int64_t> o = at::native::conv_output_size(
                input.sizes(), weight.sizes(), padding, stride, dilation);
            push(
                stack,
                at::native::empty_zendnn(
                    o,
                    optTypeMetaToScalarType(input.options().dtype_opt()),
                    input.options().layout_opt(),
                    input.options().device_opt(),
                    input.options().pinned_memory_opt()));
            return;
          }
          // aten::convolution also checks dtype mismatches
          TORCH_CHECK(
              input.options().type_equal(weight.options()),
              "Input type (",
              input.toString(),
              ") and weight type (",
              weight.toString(),
              ") should be the same");

          push(
              stack,
              at::native::zendnn_convolution_relu(
                  input, weight, bias, padding, stride, dilation, groups));
        },
        AliasAnalysisKind::FROM_SCHEMA),
});

using elemwise_fusion_patterns = std::vector<std::tuple<std::string, std::string,  std::string , std::string>>;

//fuse_elewise_ops : Clubbed all elementwise op fusions into one single function as they have lot similarity in the code flow. This will
//enable ease of integration of any elementwise fusion and speed up the fusion JIT pass

void fuse_elewise_ops(Block* b, std::function<void(Node *)> add_to_nodes_to_remove, elemwise_fusion_patterns fuse_nodes) {

  for (Node* n : b->nodes()) {
    for (Block* block : n->blocks()) {
      fuse_elewise_ops(block, add_to_nodes_to_remove, fuse_nodes);
    }

    for(auto &pattern : fuse_nodes)
    {
        if ((n->kind().toQualString() == std::get<0>(pattern) || (n->kind().toQualString() == std::get<1>(pattern))) && n->inputs().at(0)->node()->kind().toQualString() == std::get<2>(pattern))
        {
          auto prior_op = n->inputs().at(0)->node();
          auto ele_op = n;

          // check if prior_op output is used only by ele_op
          if (prior_op->output()->uses().size() > 1) {
            continue;
          }

          // Bypass ele_op and connect prior_op output directly
          ele_op->output()->replaceAllUsesWith(prior_op->output());

          // replace prior_op node with prior_op+ele_op node
          prior_op->replaceWithNewSymbol(Symbol::prim(std::get<3>(pattern)));

          // delete the old prior_op node
          prior_op->destroy();

          // as ele_op_ has aliasing relationship between input and output, it wont
          // deleted by EliminateDeadCode call. So removing those explicitly
          // after all instances of prior_op+ele_op are fused
          add_to_nodes_to_remove(ele_op);
        }
    }
  }
  return;
}


void zendnn_elmnt_op_fusion(std::shared_ptr<Graph>& graph, elemwise_fusion_patterns pattern) {
  // List of nodes to be removed explicitly
  std::vector<Node*> nodes_to_remove;

  if(pattern.empty())
  {
    return;
  }
  // fusion function which will fuse instances of zendnn_convolution
  // and relu sequences of the graph recursively
   fuse_elewise_ops(graph->block(), [&](Node *node){
    TORCH_INTERNAL_ASSERT(!node->hasUses());
    nodes_to_remove.push_back(node);
    }, pattern
  );

  // delete nodes which are added for deletion
  for (Node* node : nodes_to_remove) {
    node->destroy();
  }

  // elminate any dead code(nodes)
  EliminateDeadCode(graph);
}

#if AT_ZENDNN_QUANT_ENABLED()

size_t findArgument(
    const FunctionSchema& the_schema,
    const std::string& unqualName) {
  for (const auto i : c10::irange(the_schema.arguments().size())) {
    const Argument* arg = &the_schema.arguments()[i];
    if (arg->name() == unqualName) {
      return i;
    }
  }
  throw std::runtime_error(
      std::string("Couldn't find an argument called ") + unqualName);
}

bool fuse_vitis_ai_ops(Block* b, std::function<void(Node *)> add_to_nodes_to_remove, Value* fuse_val,
                      bool conv_add_fusion) {
  //Iterate through every node in every block of input graph

  //Traversing every node in block
  for (Node* n : b->nodes()) {
    //Traversing every block of graph
    for (Block* block : n->blocks()) {
      fuse_vitis_ai_ops(block, add_to_nodes_to_remove, fuse_val, conv_add_fusion);
    }
    Node *body_node = n;
    if (conv_add_fusion && (body_node->kind() == aten::add_ || body_node->kind() == aten::add)
     && (body_node->inputs().at(0)->node()->kind() == prim::BroadcastZENDNNTensors))
    {

      //Removing Broadcasting node, assuming no hazards for particular models
      auto brdcast_node = body_node->inputs().at(0)->node();

      //Next node information
      auto conv = brdcast_node->inputs().at(0)->node();
      if(conv->kind() != prim::zendnn_vitisai_convolution)
      {
         continue;
      }

      // check if convlution output is used only by add
      if (conv->output()->uses().size() > 1) {
         continue;
       }

      auto add_arg_2 = brdcast_node->inputs().at(1)->node();
      auto add  = body_node;

      //Replacing broadcast tensors with normal tensors in aten::add node
      body_node->replaceInputWith(body_node->inputs().at(0), brdcast_node->inputs().at(0));
      body_node->replaceInputWith(body_node->inputs().at(1), brdcast_node->inputs().at(1));

      Value* add_input =  conv->namedInput("add_input");
      conv->replaceInputWith(add_input, brdcast_node->inputs().at(1)->node()->output());

      add->output()->replaceAllUsesWith(conv->output());
      bool bf_af = add_arg_2->isAfter(conv);
      if(bf_af)
      {
        conv->moveAfter(add_arg_2);
      }

      auto conv_add_rel = body_node->next();
      if((conv_add_rel->kind() == aten::relu || conv_add_rel->kind() == aten::relu_))
      {
        int pos = findArgument(conv->schema(), "fuse_relu");
        conv->replaceInput(pos, fuse_val);

        conv_add_rel->output()->replaceAllUsesWith(conv->output());
        conv->replaceWithNewSymbol(Symbol::prim("zendnn_vitisai_convolution_add_relu"));
        add_to_nodes_to_remove(conv_add_rel);
        add_to_nodes_to_remove(add);
      }
      else
      {
        conv->replaceWithNewSymbol(Symbol::prim("zendnn_vitisai_convolution_add"));
        add_to_nodes_to_remove(add);
      }

      // delete the old conv node
      conv->destroy();
      brdcast_node->destroy();
      continue;
    }

    if ((body_node->kind() == aten::relu_ || body_node->kind() == aten::relu)
     && body_node->inputs().at(0)->node()->kind() == prim::zendnn_vitisai_convolution)
    {
      auto conv = body_node->inputs().at(0)->node();
      auto relu = body_node;
      // check if convlution output is used only by relu
      if (conv->output()->uses().size() > 1) {
        continue;
      }

      int pos = findArgument(conv->schema(), "fuse_relu");
      conv->replaceInput(pos, fuse_val);

      // Bypass relu and connect conv output directly
      relu->output()->replaceAllUsesWith(conv->output());

      // replace conv node with conv+relu node
      conv->replaceWithNewSymbol(Symbol::prim("zendnn_vitisai_convolution_relu"));

      // delete the old conv
      conv->destroy();
      add_to_nodes_to_remove(relu);
      continue;
    }
  }
  return false;
}

void zendnn_vitis_ai_fusions(std::shared_ptr<Graph>& graph)
{
  bool conv_add_fusion = zendnn::zendnn_getenv_int("ZENDNN_PT_CONV_ADD_FUSION_SAFE", 0);
  //Tracking input tensor Node of graph
  Node *node = nullptr;
  for(Value *v :  graph->inputs())
  {
     if(v->type()->cast<TensorType>())
     {
        node = graph->inputs().at(1)->uses().at(0).user;
        break;
     }
  }

  if(node == NULL)
  {
    printf("Graph does not contain input tensor");
    exit(1);
  }

  //Inserting Graph variable(Value) and set it true. Value can be used to modify schema
  //at appropriate fusion calls
  WithInsertPoint insert_val(node);
  Value* fuse_val = node->owningGraph()->insertConstant(true);

  // List of nodes to be removed explicitly
  std::vector<Node*> nodes_to_remove;

  // fusion function which will fuse instances of zendnn_convolution
  // and relu sequences of the graph recursively
  fuse_vitis_ai_ops(graph->block(), [&](Node *node){
    TORCH_INTERNAL_ASSERT(!node->hasUses());
    nodes_to_remove.push_back(node);
  }, fuse_val, conv_add_fusion);

  // delete nodes which are added for deletion
  for (Node* node : nodes_to_remove) {
    node->destroy();
  }

  //Verifying topology of nodes and values
  graph->lint();

  // elminate any dead code(nodes)
  EliminateDeadCode(graph);
}
#endif


}

void OptimizeFrozenZendnnGraph(std::shared_ptr<Graph>& graph) {
    ConvertFrozenOpsToZENDNN(graph);
#if AT_ZENDNN_QUANT_ENABLED()
    zendnn_vitis_ai_fusions(graph);
#endif
    // List of nodes to be removed explicitly
    std::vector<Node*> nodes_to_remove;

    //Vector of tuples which holds fusion patterns
    elemwise_fusion_patterns pattern;

    //push string patterns and replacement string in tuple into vector
    pattern.push_back(std::make_tuple("aten::relu", "aten::relu_", "prim::zendnn_convolution", "zendnn_convolution_relu"));

    pattern.push_back(std::make_tuple("aten::gelu", "aten::gelu_", "prim::zendnn_linear", "zendnn_linear_gelu"));
    zendnn_elmnt_op_fusion(graph, pattern);
}
#else
void OptimizeFrozenZendnnGraph(std::shared_ptr<Graph>& graph) {
    GRAPH_DUMP("ZENDNN Not enabled", graph);
}
#endif
} // namespace jit
} // namespace torch
