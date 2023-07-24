/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

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
#include <ATen/native/zendnn/Utils.h>
#include <ATen/native/zendnn/ZENDNNCommon.h>
#include <adeep.hpp>
#endif

// clang-format on

namespace torch {
namespace jit {

#if AT_ZENDNN_ENABLED()

using Tensor = at::Tensor;

namespace {

c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return AliasAnalysisKind::FROM_SCHEMA;
}

using ValueSet = std::unordered_set<Value*>;
using ValueSetPtr = std::shared_ptr<std::unordered_set<Value*>>;

Node* getLastUse(Value* v) {
  auto last_use_node = v->node();
  for (const auto& use : v->uses()) {
    if (use.user->isAfter(last_use_node)) {
      last_use_node = use.user;
    }
  }
  return last_use_node;
}

void merge_sets(
    std::unordered_map<Value*, ValueSetPtr>& alias_mapping,
    Value* existing,
    Value* new_v) {
  if (alias_mapping[existing] == alias_mapping[new_v]) {
    return;
  }
  auto existing_set = alias_mapping[existing];
  auto set_to_remove = alias_mapping[new_v];
  for (auto it = set_to_remove->begin(); it != set_to_remove->end(); it++) {
    existing_set->insert(*it);
    alias_mapping[*it] = existing_set;
  }
}

// no uses of tensors in container types
void assertNonTensorTypeDoesNotContainTensors(TypePtr type) {
  if (type->cast<TensorType>()) {
    return;
  }
  // NOLINTNEXTLINE(performance-for-range-copy)
  for (auto t : type->containedTypes()) {
    TORCH_INTERNAL_ASSERT(!t->cast<TensorType>());
  }
}

void InplaceZENDNNSubgraph(std::shared_ptr<Graph> graph) {
  // This function first calculates aliasing sets,
  // then calculates the last node each aliasing set is alive for.
  // Then we go through each node, if it's a node which has an equivalent
  // inplace node and the aliasing set for its input is dead afer this node, we
  // inplace it. Then we merge the aliasing sets for the input and output of the
  // node and extend the liveness of the set. To inplace a node you need to
  // prove device and dtype of the input and output are the same, which we've
  // already done, and prove that the output size is the same as the input size,
  // which is achieved by explicit Broadcast nodes (which we inserted for other
  // reasons).
  // The graphs here are simple subgraphs without uses of Tensors in
  // containers (Lists, GetAttrs, etc)

  // CALCULATE ALIASING SETS

  auto aliasDb = torch::make_unique<AliasDb>(graph);

  // map from Value to its Aliasing Set
  std::unordered_map<Value*, ValueSetPtr> alias_mapping;
  ValueSet set;
  ValueSetPtr input_set = std::make_shared<ValueSet>(set);
  for (Value* v : graph->inputs()) {
    if (v->type()->cast<TensorType>()) {
      input_set->insert(v);
      alias_mapping[v] = input_set;
    } else {
      assertNonTensorTypeDoesNotContainTensors(v->type());
    }
  }

  for (Node* n : graph->nodes()) {
    for (Value* output : n->outputs()) {
      if (!output->type()->cast<TensorType>() && !output->type()->cast<ListType>()) {
        assertNonTensorTypeDoesNotContainTensors(output->type());
        continue;
      }

      std::unordered_set<Value*> new_set = {output};
      alias_mapping[output] = std::make_shared<ValueSet>(new_set);
      for (Value* input : n->inputs()) {
        if (aliasDb->mayAlias(input, output)) {
          merge_sets(alias_mapping, input, output);
        }
      }
    }
  }

  // CALCULATE ALIASING SET LIVENESS

  // map from aliased set -> last use of set
  std::unordered_map<ValueSetPtr, Node*> set_liveness;
  for (auto& set : alias_mapping) {
    if (set_liveness.count(set.second)) {
      continue;
    }
    Node* last = nullptr;
    // NOLINTNEXTLINE(modernize-loop-convert)
    for (auto it = set.second->begin(); it != set.second->end(); it++) {
      Value* v = *it;
      auto k = v->node()->kind();
      if (k == prim::Constant || k == prim::ConstantZENDNNTensor ||
          k == prim::Param) {
        last = graph->return_node();
        continue;
      }

      auto last_use = getLastUse(v);
      if (!last || last_use->isAfter(last)) {
        last = last_use;
      }
    }
    set_liveness[set.second] = last;
  }

  // REUSING MEMORY BY REINPLACING NODES
  std::vector<Node*> nodes_to_inplace;

  auto add_to_inplace_set = [&](Node* node) {
    // defer making the inplacing change because that would invalidate the old
    // Node output Value*
    nodes_to_inplace.push_back(node);
    TORCH_INTERNAL_ASSERT(node->outputs().size() == 1);
    auto output_liveness_end =
        set_liveness[alias_mapping[node->outputs().at(0)]];
    merge_sets(alias_mapping, node->inputs().at(0), node->output());
    set_liveness[alias_mapping[node->output()]] = output_liveness_end;
  };

  for (Node* node : graph->nodes()) {
    auto k = node->kind();
    if (k == aten::relu || k == aten::sigmoid || k == aten::dropout ||
        k == prim::ZENDNNHardSwish || k == prim::ZENDNNHardSigmoid ||
        k == prim::ZENDNNHardTanh || k == aten::tanh ||
        k == prim::ZENDNNClamp || k == Symbol::prim("ZENDNNLayerNorm")) {
      if (set_liveness[alias_mapping[node->inputs().at(0)]]->isAfter(node)) {
        continue;
      }
      add_to_inplace_set(node);
    } else if (k == aten::mul || k == aten::add) {
      // the binary operators (add/mul) are commutative and only take tensor
      // inputs, so we can inplace either the first or second input
      int64_t reusable_value_index = -1;
      for (size_t i = 0; i < 2; i++) {
        TORCH_INTERNAL_ASSERT(node->inputs().at(i)->type()->cast<TensorType>());
        if (!set_liveness[alias_mapping[node->inputs().at(i)]]->isAfter(node)) {
          reusable_value_index = i;
          break;
        }
      }

      if (reusable_value_index == -1) {
        continue;
      }

      if (reusable_value_index == 1) {
        node->insertInput(0, node->inputs().at(1));
        node->removeInput(2);
      }
      add_to_inplace_set(node);
    }
  }

  for (Node* node : nodes_to_inplace) {
    node->replaceWithNewSymbol(
        Symbol::fromQualString(node->schema().name() + "_"));
    node->destroy();
  }
}

// This is a factory function that creates an Operation that that takes
// ZENDNN tensors and unpacks them into 1D contiguous tensors that we can
// run aten operations on. The precondition for using this function is that the
// aten operations in `aten_op` should be an identity for zero inputs. In other
// words, this should: `aten_op(0) = 0` The reason for this precondition has to
// do with blocked formats ZENDNN uses to lay tensor elements (nChw8c, nChw16c,
// etc). It splits the channel dimension into chunks of 8/16 makes it the
// innermost dimension. Whenever the channel dim isn't divisible by 8/16 the
// innermost dimension is padded with 0s. The precondition, `aten_op(0) == 0`
// allows us to avoid any special casing of padded elements.
Operation createUnaryOp(
    std::function<void(at::Tensor output, at::Tensor input)> aten_op,
    bool inplace = false) {
  return [aten_op, inplace](Stack* stack) {
    auto a = pop(stack).toTensor();
    c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
    // we cast `a` to an `adeep::tensor`, so we can get at its descriptor
    // which we then use to set up `out` tensor w/ the same props as a
    auto a_it = at::native::itensor_from_zendnn(a);
    auto zendnn_raw_data = a_it.get_data_handle();
    auto a_options_with_strided = a.options().layout(c10::kStrided);
    // we also wrap `a` storage into an aten tensor
    auto in_aten =
        at::from_blob(zendnn_raw_data, {a.numel()}, a_options_with_strided);

    auto out_raw_data = zendnn_raw_data;
    auto out = a;
    if (!inplace) {
      // `a_it.get_desc()` will allocate a tensor
      // of the right physical size.
      auto it_empty = adeep::tensor(a_it.get_desc());
      TORCH_INTERNAL_ASSERT(it_empty.get_desc() == a_it.get_desc());
      out = at::native::new_with_itensor_zendnn(
          std::move(it_empty),
          optTypeMetaToScalarType(a.options().dtype_opt()),
          a.options().device_opt());

      out_raw_data = at::native::itensor_from_zendnn(out).get_data_handle();
    }

    // tensor's physical size could be bigger than a logical one
    // `a_it.get_desc().get_size()` returns the real physical size (in bytes)
    // we use it to compute `nelem` for `aten` ops
    TORCH_INTERNAL_ASSERT(
        a_it.get_desc().get_size() % elementSize(a.scalar_type()) == 0);
    auto nelem = a_it.get_desc().get_size() / elementSize(a.scalar_type());
    auto out_aten = at::from_blob(
        out_raw_data, {static_cast<int64_t>(nelem)}, a_options_with_strided);
    aten_op(out_aten, in_aten);
    push(stack, out);
  };
}

void ZENDNNLayerNormOp(Stack& stack, bool inplace) {
  c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);

  // enable_cudnn not used
  pop(stack);
  auto eps = pop(stack).toDouble();

  Tensor bias{};
  Tensor weight{};
  auto bias_ival = pop(stack);
  TORCH_INTERNAL_ASSERT(bias_ival.isTensor());
  bias = bias_ival.toTensor();

  auto weight_ival = pop(stack);
  TORCH_INTERNAL_ASSERT(weight_ival.isTensor());
  weight = weight_ival.toTensor();

  auto shape = pop(stack).toDimVector();
  auto input = pop(stack).toTensor();

  at::Tensor dst, mean, rstd;
  std::tie(dst, mean, rstd) =
      at::native::zendnn_layer_norm_last_index_weight_bias_f32(
          input, shape, weight, bias, eps, inplace);
  push(stack, dst);
};

Operation BroadOp(const Node* node) {
  return [](Stack* stack) {
    auto b = pop(stack).toTensor();
    auto a = pop(stack).toTensor();
    auto b_size = b.sizes();
    auto a_size = a.sizes();
    if (a_size.equals(b_size)) {
      // TODO: follow up with ZENDNN what the best way is
      // to handle perf incompatible formats
      push(stack, a, b);
      return;
    } else {
      auto out_size = at::infer_size(a_size, b_size);
      int64_t out_numel = out_size[0];
      for (size_t i = 1, end = out_size.size(); i < end; ++i) {
        out_numel = out_numel * out_size[i];
      }

      auto exp_a = a;
      auto exp_b = b;
      // zendnn tensors only support reshape, not expand or view operators
      if (a_size.equals(out_size)) {
        push(stack, a);
      } else if (out_numel == a.numel()) {
        exp_a = a.reshape(out_size);
      } else {
        // TODO: consider to initializing to a blocked layout
        // directly if needed
        exp_a = a.to_dense().expand(out_size).to_zendnn();
      }

      if (b_size.equals(out_size)) {
        push(stack, b);
      } else if (out_numel == b.numel()) {
        exp_b = b.reshape(out_size);
      } else {
        exp_b = b.to_dense().expand(out_size).to_zendnn();
      }

      {
        // If one of the inputs was expanded and converted to nchw/nhwc
        // we might end up in a very bad spot if the second argument
        // is in a blocked format. In this case, ZENDNN uses its
        // reference implementation for a binary operation that follows
        // these broadcasts and it could be up to ~100x slower.
        // We use a very simple heuristic to convert an arg in nchw
        // to the blocked format of the other argument.
        c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
        auto a_it = at::native::itensor_from_zendnn(exp_a);
        auto b_it = at::native::itensor_from_zendnn(exp_b);

        // `is_public_format` means a tensor's physical layout isn't in ZENDNN
        // blocked layout e.g. nchw or nhwc but not nChw8c
        if (!a_it.is_public_format()) {
          if (b_it.is_public_format()) {
            b_it = b_it.reorder_if_differ_in(a_it.get_desc());
          }
        } else if (!b_it.is_public_format()) {
          if (a_it.is_public_format()) {
            a_it = a_it.reorder_if_differ_in(b_it.get_desc());
          }
        }

        auto a_options = exp_a.options();
        auto a_out = at::native::new_with_itensor_zendnn(
            std::move(a_it),
            optTypeMetaToScalarType(a_options.dtype_opt()),
            a_options.device_opt());
        push(stack, a_out);
        auto b_options = exp_b.options();
        auto b_out = at::native::new_with_itensor_zendnn(
            std::move(b_it),
            optTypeMetaToScalarType(b_options.dtype_opt()),
            b_options.device_opt());
        push(stack, b_out);
      };
    }
  };
}

static std::function<void(at::Tensor output, at::Tensor input)> hardtanh_helper(
    const Node* n) {
  auto min_val = n->f(attr::min_val);
  auto max_val = n->f(attr::max_val);
  return [min_val, max_val](at::Tensor output, at::Tensor input) {
    at::cpu::hardtanh_out(output, input, min_val, max_val);
  };
}

static std::function<void(at::Tensor output, at::Tensor input)> clamp_helper(
    const Node* n) {
  auto min_val = n->f(attr::min_val);
  auto max_val = n->f(attr::max_val);
  return [min_val, max_val](at::Tensor output, at::Tensor input) {
    at::cpu::clamp_out(output, input, min_val, max_val);
  };
}

// any op added to this registry needs to meet
// the precondition: `aten_op(0) == 0`
const RegisterOperators ZENDNNHardSwishOpReg({
    torch::jit::Operator(
        "prim::ZENDNNHardSwish_(Tensor(a!) self) -> Tensor(a!)",
        createUnaryOp(
            [](at::Tensor output, at::Tensor input) {
              at::cpu::hardswish_out(output, input);
            },
            true),
        AliasAnalysisKind::FROM_SCHEMA),
    torch::jit::Operator(
        "prim::ZENDNNHardSigmoid_(Tensor(a!) self) -> Tensor(a!)",
        createUnaryOp(
            [](at::Tensor output, at::Tensor input) {
              at::cpu::hardsigmoid_out(output, input);
            },
            true),
        AliasAnalysisKind::FROM_SCHEMA),
    torch::jit::Operator(
        "prim::ZENDNNHardTanh_(Tensor(a!) self) -> Tensor(a!)",
        [](const Node* n) -> Operation {
          return createUnaryOp(hardtanh_helper(n), true);
        },
        AliasAnalysisKind::FROM_SCHEMA),
    torch::jit::Operator(
        "prim::ZENDNNClamp_(Tensor(a!) self) -> Tensor(a!)",
        [](const Node* n) -> Operation {
          return createUnaryOp(clamp_helper(n), true);
        },
        AliasAnalysisKind::FROM_SCHEMA),
    torch::jit::Operator(
        "prim::ZENDNNHardSwish(Tensor a) -> Tensor",
        createUnaryOp(
            [](at::Tensor output, at::Tensor input) {
              at::cpu::hardswish_out(output, input);
            },
            false),
        AliasAnalysisKind::FROM_SCHEMA),
    torch::jit::Operator(
        "prim::ZENDNNHardSigmoid(Tensor a) -> Tensor",
        createUnaryOp(
            [](at::Tensor output, at::Tensor input) {
              at::cpu::hardsigmoid_out(output, input);
            },
            false),
        AliasAnalysisKind::FROM_SCHEMA),
    torch::jit::Operator(
        "prim::ZENDNNHardTanh(Tensor self) -> Tensor",
        [](const Node* n) -> Operation {
          return createUnaryOp(hardtanh_helper(n), false);
        },
        AliasAnalysisKind::FROM_SCHEMA),
    torch::jit::Operator(
        "prim::ZENDNNClamp(Tensor self) -> Tensor",
        [](const Node* n) -> Operation {
          return createUnaryOp(clamp_helper(n), false);
        },
        AliasAnalysisKind::FROM_SCHEMA),
});

const RegisterOperators BroadOpReg({
    torch::jit::Operator(
        prim::BroadcastZENDNNTensors,
        BroadOp,
        AliasAnalysisKind::INTERNAL_SPECIAL_CASE),
});

const RegisterOperators ZENDNNLayerNormOpReg({
    torch::jit::Operator(
        "prim::ZENDNNLayerNorm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=False) -> Tensor",
        [](Stack& stack) { ZENDNNLayerNormOp(stack, false); },
        AliasAnalysisKind::FROM_SCHEMA),
    torch::jit::Operator(
        "prim::ZENDNNLayerNorm_(Tensor(a!) input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=False) -> Tensor(a!)",
        [](Stack& stack) { ZENDNNLayerNormOp(stack, true); },
        AliasAnalysisKind::FROM_SCHEMA),
});

Operation ConstantZENDNNTensorOp(const Node* node) {
  const auto& t = node->t(attr::value);
  return [t](Stack* stack) {
    push(stack, t);
    return 0;
  };
}

// aten::convolution does a lot of precomputation and dispatching before
// zendnn_convolution is called. registering here we can directly invoke the op
// and avoid overhead. avoiding dispatch overhead for other operators - relu,
// add, etc - did not benchmark as speeding up models noticeably. the additional
// overhead of `convolution` warrants the custom operator.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
jit::RegisterOperators reg_fut_ops({
    jit::Operator(
        // XXX: this follows the schema convention of conv2d/conv3d, not
        // aten::zendnn_convolution, which is different for some reason!
        "prim::zendnn_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor",
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
              at::native::zendnn_convolution(
                  input, weight, bias, padding, stride, dilation, groups));
        },
        aliasAnalysisFromSchema()),
});

jit::RegisterOperators reg_zendnn_linear({
    jit::Operator(
        // XXX: this follows the schema convention of conv2d/conv3d, not
        // aten::zendnn_linear, which is different for some reason!
        "prim::zendnn_linear(Tensor input, Tensor weight, Tensor? bias) -> Tensor",
        [](jit::Stack* stack) {
          Tensor bias;
          IValue bias_ival = pop(stack);
          if (!bias_ival.isNone()) {
            bias = bias_ival.toTensor();
          }
          Tensor weight = pop(stack).toTensor();
          Tensor input = pop(stack).toTensor();

          at::AutoDispatchBelowAutograd mode;

          // aten::linear also checks dtype mismatches
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
                  input, weight, bias, false));
        },
        aliasAnalysisFromSchema()),
});

#if AT_ZENDNN_QUANT_ENABLED()
Operation ZenDNN_Vitis_Ai_Conv_Op(const Node* node) {
  return [](jit::Stack* stack) {
          int64_t add_out_scale = pop(stack).toInt();
          int64_t add_scale = pop(stack).toInt();
          int64_t output_scale = pop(stack).toInt();
          int64_t filter_scale = pop(stack).toInt();
          int64_t input_scale = pop(stack).toInt();
          bool fuse_relu = pop(stack).toBool();
          bool dequant = pop(stack).toBool();

          int64_t groups = pop(stack).toInt();
          auto dilation = pop(stack).toIntVector();
          auto stride = pop(stack).toIntVector();
          auto padding = pop(stack).toIntVector();
          Tensor bias;
          Tensor add_input;
          IValue add_input_ival = pop(stack);
          if (!add_input_ival.isNone()) {
            add_input = add_input_ival.toTensor();
          }

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
            std::vector<int64_t> out = at::native::conv_output_size(
                input.sizes(), weight.sizes(), padding, stride, dilation);
            push(
                stack,
                at::native::empty_zendnn(
                    out,
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
              at::native::zendnn_vitisai_convolution(
                  input, weight, bias, add_input, padding, stride, dilation, groups, dequant, fuse_relu, input_scale, filter_scale, output_scale,add_scale,add_out_scale));
        };
}

const RegisterOperators ZENDNNVitisAiFuseOpReg({
    torch::jit::Operator(
        "prim::zendnn_vitisai_convolution(Tensor self, Tensor weight, Tensor? bias, Tensor? add_input, int[] padding, int[] stride, int[] dilation, int groups, bool dequant, bool fuse_relu, int input_scale, int filter_scale, int output_scale,int add_scale, int add_out_scale) -> Tensor",
        ZenDNN_Vitis_Ai_Conv_Op,
        AliasAnalysisKind::FROM_SCHEMA),

    torch::jit::Operator(
        "prim::zendnn_vitisai_convolution_add(Tensor self, Tensor weight, Tensor? bias, Tensor? add_input, int[] padding, int[] stride, int[] dilation, int groups, bool dequant, bool fuse_relu, int input_scale, int filter_scale, int output_scale,int add_scale, int add_out_scale) -> Tensor",
        ZenDNN_Vitis_Ai_Conv_Op,
        AliasAnalysisKind::FROM_SCHEMA),

    torch::jit::Operator(
        "prim::zendnn_vitisai_convolution_relu(Tensor self, Tensor weight, Tensor? bias, Tensor? add_input, int[] padding, int[] stride, int[] dilation, int groups, bool dequant, bool fuse_relu, int input_scale, int filter_scale, int output_scale, int add_scale, int add_out_scale) -> Tensor",
        ZenDNN_Vitis_Ai_Conv_Op,
        AliasAnalysisKind::FROM_SCHEMA),

    torch::jit::Operator(
        "prim::zendnn_vitisai_convolution_add_relu(Tensor self, Tensor weight, Tensor? bias, Tensor? add_input, int[] padding, int[] stride, int[] dilation, int groups, bool dequant, bool fuse_relu, int input_scale, int filter_scale, int output_scale, int add_scale, int add_out_scale) -> Tensor",
        ZenDNN_Vitis_Ai_Conv_Op,
        AliasAnalysisKind::FROM_SCHEMA),
});
#endif
// This is registered as its own op instead of as prim::Constant bc it does not
// serialize which is an invariant of prim::Constant
// TODO: make zendnn tensor serialize...
const RegisterOperators ZENDNNConstantOp({
    torch::jit::Operator(
        prim::ConstantZENDNNTensor,
        ConstantZENDNNTensorOp,
        AliasAnalysisKind::INTERNAL_SPECIAL_CASE),
});

Node* createConstantZENDNNTensorOp(Graph* g, const Tensor& zendnn_tensor) {
  TORCH_INTERNAL_ASSERT(zendnn_tensor.is_zendnn());
  auto op = g->create(prim::ConstantZENDNNTensor);
  op->t_(attr::value, zendnn_tensor);
  return op;
}

bool supportedZENDNNWeight(const Tensor& weight) {
  return weight.device().is_cpu() && (weight.dtype() == c10::ScalarType::Float || weight.dtype() == c10::ScalarType::BFloat16) &&
      weight.ndimension() != 0;
}

void replaceInputWithZENDNNTensor(Node* n, size_t index) {
  Value* input = n->inputs().at(index);
  auto zendnn_tensor = constant_as<Tensor>(input)->to_zendnn();
  auto zendnn_tensor_value =
      createConstantZENDNNTensorOp(n->owningGraph(), zendnn_tensor)
          ->insertBefore(n)
          ->output();
  zendnn_tensor_value->setDebugName(input->debugName() + "_zendnn");
  n->replaceInputWith(input, zendnn_tensor_value);
}

void replaceInputWithZENDNNTensor(
    Node* n,
    const std::string& name,
    const at::Tensor& zendnn_tensor) {
  Value* input = n->namedInput(name);
  auto zendnn_tensor_value =
      createConstantZENDNNTensorOp(n->owningGraph(), zendnn_tensor)
          ->insertBefore(n)
          ->output();
  zendnn_tensor_value->setDebugName(input->debugName() + "_zendnn");
  n->replaceInputWith(input, zendnn_tensor_value);
}

void replaceInputWithZENDNNTensor(Node* n, const std::string& name) {
  Value* input = n->namedInput(name);
  auto zendnn_tensor = constant_as<Tensor>(input)->to_zendnn();
  replaceInputWithZENDNNTensor(n, name, zendnn_tensor);
}

void moveConvWeightsToZENDNN(Node* conv) {
  static bool first_conv = false;
  auto conv_w_zendnn =
      constant_as<Tensor>(conv->namedInput("weight")).value().to_zendnn();
  std::vector<int64_t> padding =
      toIValue(conv->namedInput("padding"))->toIntVector();
  std::vector<int64_t> stride =
      toIValue(conv->namedInput("stride"))->toIntVector();
  std::vector<int64_t> dilation =
      toIValue(conv->namedInput("dilation"))->toIntVector();
  auto groups = constant_as<int64_t>(conv->namedInput("groups")).value();

  if (conv->kind() == aten::conv2d) {
    conv_w_zendnn = zendnn_reorder_conv2d_weight(
        conv_w_zendnn, padding, stride, dilation, groups);
  } else if (conv->kind() == aten::conv3d) {
    conv_w_zendnn = zendnn_reorder_conv3d_weight(
        conv_w_zendnn, padding, stride, dilation, groups);
  }
#if AT_ZENDNN_QUANT_ENABLED()
  else if (conv->kind() == aten::zendnn_vitisai_convolution) {
    int64_t wgt_scale =constant_as<int64_t>(conv->namedInput("filter_scale")).value();
    Node *temp = conv->inputs().at(0)->node();
    bool u8_inp = false;
    //The first convolution node(feeds by fp32 data type) in block expects s8 format weights. prim::Param represents input of the
    //block. Spotting the first convolution using prim::param in the block
    while(temp->kind() != prim::Param && temp->kind() != prim::zendnn_vitisai_convolution && temp->kind() != prim::zendnn_convolution
         && temp->inputs().at(0)->node()->kind() != prim::zendnn_convolution && temp->kind() != aten::zendnn_vitisai_convolution)
    {
      //Tracking Input Data type
      if(temp->kind()==aten::relu || temp->kind() == aten::relu_)
      {
        u8_inp = true;
        break;
      }
      temp = temp->inputs().at(0)->node();
    }
    conv_w_zendnn = zendnn_reorder_conv2d_vitisai_weight(
        conv_w_zendnn, padding, stride, dilation, groups, wgt_scale, u8_inp);
  }
#endif
   else {
    TORCH_INTERNAL_ASSERT(false);
  }
  replaceInputWithZENDNNTensor(conv, "weight", conv_w_zendnn);

  if (conv->namedInput("bias")->type() != NoneType::get()) {
    replaceInputWithZENDNNTensor(conv, "bias");
  }
}

void moveWeightsToZENDNN(Node* n) {
  // conv goes through special pathway so we can call zendnn reorder conv
  // primitive
  if (n->kind() == aten::conv2d || n->kind() == aten::conv3d
#if AT_ZENDNN_QUANT_ENABLED()
  || n->kind() == aten::zendnn_vitisai_convolution
#endif
) {
    moveConvWeightsToZENDNN(n);
  } else {
    for (size_t i = 0; i < n->inputs().size(); ++i) {
      if (!n->input(i)->type()->cast<TensorType>() ||
          n->input(i)->node()->kind() != prim::Constant) {
        continue;
      }
      replaceInputWithZENDNNTensor(n, i);
    }
  }
}

static void clamp_node_creator(
    Node* body_node,
    c10::Symbol kind,
    double min_val,
    double max_val) {
  WithInsertPoint insert_guard{body_node};
  auto out_node =
      body_node->owningGraph()->create({kind}, {body_node->input(0)}, 1);
  // N.B. we can't use `insert` as it calls `getOperation` (via
  // `emitBuiltinCall`) which uses `min_val` and `max_val` attrs which we
  // haven't set yet.
  body_node->owningGraph()->insertNode(out_node);
  auto out_val = out_node->output();
  out_node->f_(attr::min_val, min_val);
  out_node->f_(attr::max_val, max_val);
  out_val->copyMetadata(body_node->output());
  body_node->output()->replaceAllUsesWith(out_val);
  body_node->destroy();
}

void ComputeSubgraphInZENDNN(Node* subgraph_node) {
  auto graph = subgraph_node->owningGraph();
  Value* none_value = nullptr;
  {
    WithInsertPoint guard(subgraph_node);
    none_value = graph->insertConstant(IValue());
  }
  for (size_t i = 0; i < subgraph_node->inputs().size(); ++i) {
    Value* v = subgraph_node->inputs().at(i);
    if (!v->type()->cast<TensorType>()) {
      continue;
    }
    auto to_zendnn =
        graph->create(c10::Symbol::fromQualString("aten::to_zendnn"), 1)
            ->insertBefore(subgraph_node);
    to_zendnn->addInput(v);
    to_zendnn->addInput(none_value);
    subgraph_node->replaceInput(i, to_zendnn->output());
  }

  for (size_t i = 0; i < subgraph_node->outputs().size(); ++i) {
    Value* v = subgraph_node->outputs().at(i);
    if (v->type()->cast<ListType>()) {
      auto from_zendnn_1 =
          graph
          ->create(
              Symbol::aten("to_dense_list"), { v, none_value })
           ->insertAfter(subgraph_node);
           at::Tensor output;
           from_zendnn_1->output()->setType(ListType::create(from_zendnn_1->outputs().at(0)->type()));
           v->replaceAllUsesAfterNodeWith(from_zendnn_1, from_zendnn_1->output());
    }
    if (!v->type()->cast<TensorType>()) {
      continue;
    }
    auto from_zendnn =
        graph
            ->create(
                c10::Symbol::fromQualString("aten::to_dense"), {v, none_value})
            ->insertAfter(subgraph_node);
    v->replaceAllUsesAfterNodeWith(from_zendnn, from_zendnn->output());
  }

  auto subgraph = SubgraphUtils::getSubgraph(subgraph_node);
  for (auto it = subgraph->block()->nodes().begin();
       it != subgraph->block()->nodes().end();) {
    Node* body_node = *it;
    it++;

    moveWeightsToZENDNN(body_node);

    if (body_node->kind() == aten::add || body_node->kind() == aten::mul) {
      auto node = body_node->owningGraph()->create(
          Symbol::prim("BroadcastZENDNNTensors"),
          {body_node->inputs().at(0), body_node->inputs().at(1)},
          2);
      node->insertBefore(body_node);
      body_node->replaceInput(0, node->outputs().at(0));
      body_node->replaceInput(1, node->outputs().at(1));
    }
    if (body_node->matches(
            "aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=False) -> Tensor")) {
      body_node->replaceWithNewSymbol(Symbol::prim("ZENDNNLayerNorm"));
      body_node->destroy();
      continue;
    }

    if (body_node->kind() == aten::hardswish) {
      body_node->replaceWithNewSymbol(prim::ZENDNNHardSwish);
      body_node->destroy();
      continue;
    }

    if (body_node->kind() == aten::hardsigmoid) {
      body_node->replaceWithNewSymbol(prim::ZENDNNHardSigmoid);
      body_node->destroy();
      continue;
    }

    if (body_node->kind() == aten::relu6) {
      clamp_node_creator(body_node, prim::ZENDNNHardTanh, 0., 6.);
      continue;
    }

    if (body_node->kind() == aten::hardtanh) {
      auto min_val =
          constant_as<double>(body_node->namedInput("min_val")).value();
      auto max_val =
          constant_as<double>(body_node->namedInput("max_val")).value();
      clamp_node_creator(body_node, prim::ZENDNNHardTanh, min_val, max_val);
      continue;
    }

    if (body_node->kind() == aten::clamp) {
      auto min_val = constant_as<double>(body_node->namedInput("min")).value();
      auto max_val = constant_as<double>(body_node->namedInput("max")).value();
      clamp_node_creator(body_node, prim::ZENDNNClamp, min_val, max_val);
      continue;
    }

    if (body_node->kind() == aten::conv2d ||
        body_node->kind() == aten::conv3d) {
      // this node doesnt handle string padding yet...
      if (!body_node->namedInput("padding")->type()->cast<StringType>()) {
        body_node->replaceWithNewSymbol(Symbol::prim("zendnn_convolution"));
        body_node->destroy();
        continue;
      }
    }

    if (body_node->kind() == aten::linear) {
        body_node->replaceWithNewSymbol(Symbol::prim("zendnn_linear"));
        body_node->destroy();
        continue;
    }

    #if AT_ZENDNN_QUANT_ENABLED()
     if (body_node->kind() == aten::zendnn_vitisai_convolution) {
      if (!body_node->namedInput("padding")->type()->cast<StringType>()) {
        body_node->replaceWithNewSymbol(Symbol::prim("zendnn_vitisai_convolution"));
        body_node->destroy();
        continue;
      }
    }
    #endif
  }
}

bool nonConstantParameters(Node* n) {
  for (size_t i = 1; i < n->inputs().size(); i++) {
    if (n->inputs().at(i)->node()->kind() != prim::Constant) {
      return true;
    }
  }
  return false;
}

bool frozenZendnnCompatibleLinearNode(Node* n) {
  if (nonConstantParameters(n)) {
    return false;
  }

  if (n->kind() != aten::linear) {
    return false;
  }

  auto weight = constant_as<Tensor>(n->namedInput("weight")).value();
  return supportedZENDNNWeight(weight);
}

bool frozenZendnnCompatibleConvNode(Node* n) {
  if (nonConstantParameters(n)) {
    return false;
  }
  // zendnn does not support conv1d
  // _convolution is rewritten before this pass is invoked
  if (n->kind() != aten::conv2d && n->kind() != aten::conv3d && n->kind() != aten::zendnn_vitisai_convolution) {
    return false;
  }

  auto weight = constant_as<Tensor>(n->namedInput("weight")).value();
  return supportedZENDNNWeight(weight);
}

// [zendnn perf strategy]
// Certain ops - aten::linear, aten::conv2d, aten::conv3d - provide a huge speed
// up just by converting the constant weights to ZENDNN AOT, and then at runtime
// converting the non-constant input to_zendnn before the op, and then back to
// its original layout after the op. The speed up holds even if you end up
// converting the input to_zendnn and output back to_dense. We start groups of
// ops to compute in ZENDNN only from these ops that are a strict speedup. Then,
// we expand the groups to include operators which are computable in ZENDNN &
// are roughly perf equal to eager. We do this in the hopes of joining multiple
// fast nodes together, saving to_zendnn and to_dense conversions.
//
// ZENDNN only supports float32 inputs for aten::linear, aten::conv2d &
// aten::conv3d. We only fuse these nodes if the weights are float32, and then
// we only include operators which we can prove will execute in float32. By
// fusing topologically we can maintain the invariant that all tensor types in
// the graph are floating point. In fusing Conv-> Add -> Relu -> Conv we start
// with the first Conv, know that the output is float, and can then safely merge
// Add and Relu. If we started with the last Conv, it would be difficult to
// prove in our first pass that the Add's inputs were both float32 without first
// fusing the first conv.

class ZENDNNSubgraphSlicer {
 public:
  ZENDNNSubgraphSlicer(
      Block* block,
      std::shared_ptr<Graph> graph,
      AliasDb& aliasDb)
      : block_(block), graph_(std::move(graph)), aliasDb_(aliasDb) {}

  void run() {
    // We maintain alias db correctness in-place while building up the autodiff
    // subgraphs, however it is difficult to preserve correctness when
    // un-inlining autodiff subgraphs. We first recursively construct all
    // subgraphs and then unmerge them into the graph
    buildupSubgraphs();
    computeSubgraphsInZENDNN();
    // Run CSE globally onceto eliminate duplicates that may have occurred
    // while inlining subgraphs.
    EliminateCommonSubexpression(graph_);
  }

  void buildupSubgraphs() {
    // We need to run the slicer multiple times in order to get all merge
    // opportunities. This is because moveBeforeTopologicalValid may reorder
    // nodes to be AFTER the current iteration point. In order to properly
    // consider those nodes for merging, we need run the pass until no changes
    // have been made.
    //
    // Example:
    //   c = f(a, b)
    //   d = f(c)
    //   e = f(d)  <- iter is here, moving upward
    // After c.moveBeforeTopologicallyValid(e), we have:
    //   c = f(a, b)
    //   e = f(d)  <- iter still here
    //   d = f(c)  <- this was node moved on the other side.

    bool any_changed = true;
    while (any_changed) {
      any_changed = false;
      for (auto it = block_->nodes().begin(); it != block_->nodes().end();) {
        bool changed = false;
        std::tie(it, changed) = scanNode(*it);
        any_changed |= changed;
      }
    }

    // Construct Subgraphs Recursively
    for (Node* n : block_->nodes()) {
      for (auto subBlock : n->blocks()) {
        ZENDNNSubgraphSlicer(subBlock, graph_, aliasDb_).buildupSubgraphs();
      }
    }
  }

  static bool ZENDNNGroupStart(Node* node) {
    // if we're already in the process of merging
    if (node->kind() == prim::ZENDNNGroup) {
      return true;
    }
    // see [zendnn perf strategy]
    return frozenZendnnCompatibleLinearNode(node) ||
        frozenZendnnCompatibleConvNode(node);
  }

 private:
  // ZENDNN only supports floats of dimension > 0, so we only support
  // Tensors who have a known type or were previously verified
  // to be usable in an ZENDNN Group
  bool tensorInputIsZENDNNSupported(Value* v, Node* v_use) {
    auto const_tensor = constant_as<Tensor>(v);
    if (const_tensor) {
      return supportedZENDNNWeight(*const_tensor);
    }
    auto k = v->node()->kind();
    if (k == prim::ZENDNNGroup || k == prim::ConstantZENDNNTensor ||
        k == aten::to_zendnn) {
      return true;
    }
    for (const auto& use : v->uses()) {
      if (use.user->kind() == aten::to_zendnn &&
          v_use->owningBlock() == use.user->owningBlock() || use.user->kind() == prim::ZENDNNGroup ) {
        return true;
      }
    }
    return false;
  }

  // We include ops here which are roughly perf-equivalent in zendnn as with
  // aten (single & multithreaded) and whose inputs & outputs are float32.
  bool computableInZENDNN(Node* n) {
    for (Value* v : n->inputs()) {
      if (v->type()->cast<TensorType>() &&
          !(tensorInputIsZENDNNSupported(v, n))) {
        return false;
      }
    }

    if (n->matches(
            "aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=False) -> Tensor") &&
        n->namedInput("weight")->type() != NoneType::get() &&
        n->namedInput("bias")->type() != NoneType::get()
        ) {
      auto norm_shape =
          constant_as<std::vector<int64_t>>(n->namedInput("normalized_shape"));
      return norm_shape.has_value() && norm_shape->size() == 1;
    }

    // unary ops we dont need to prove anything else than
    // the input is zendnn supported
    switch (n->kind()) {
      case aten::cat:
      case prim::ListConstruct:
      case aten::relu:
      case aten::relu6:
      case aten::gelu:
      case aten::sigmoid:
      case aten::hardsigmoid:
      case aten::hardswish:
      case aten::tanh:
      case aten::batch_norm:
      // TODO: max_pool on zendnn can be slower than in eager. ideally, we'd
      // only fuse it if we knew including max_pool lead to fewer layout
      // conversions. from initial testing including it speeds up models
      case aten::max_pool2d:
      case aten::max_pool3d:
      case aten::avg_pool2d:
      case aten::adaptive_avg_pool2d:
      case aten::avg_pool3d:
        // case aten::adaptive_max_pool2d: // return tuples which break fusion
        // case aten::adaptive_max_pool3d: // return tuples which break fusion
        // case aten::adaptive_avg_pool3d: // no adeep binding
        return true;
    }

    if ((n->kind() == aten::hardtanh || n->kind() == aten::clamp) &&
        !nonConstantParameters(n)) {
      const size_t MIN_INDEX = 1, MAX_INDEX = 2;
      auto min_val = constant_as<double>(n->input(MIN_INDEX)).value();
      auto max_val = constant_as<double>(n->input(MAX_INDEX)).value();
      // we need to maintain the following invariant `pointwise_func(0) == 0`,
      // see `createUnaryOp`
      if (min_val <= 0. && max_val >= 0.) {
        return true;
      }
    }

    if (n->kind() == aten::add || n->kind() == aten::mul) {
      // zendnn doesn't currently support Tensor-Scalar add
      for (size_t i = 0; i < 2; i++) {
        if (!n->inputs().at(i)->type()->cast<TensorType>()) {
          return false;
        }
      }
      return true;
    }
    // TODO: dropout removal. zendnn doesnt support train=True
    if (n->kind() == aten::dropout) {
      auto train = constant_as<bool>(n->namedInput("train")).value();
      return train == false;
    }
    return false;
  }

  void computeSubgraphsInZENDNN() {
    auto curNode = *block_->nodes().begin();
    while (curNode != *block_->nodes().end()) {
      auto nextNode = curNode->next();
      if (curNode->kind() == prim::ZENDNNGroup) {
        ComputeSubgraphInZENDNN(curNode);
        InplaceZENDNNSubgraph(SubgraphUtils::getSubgraph(curNode));
        SubgraphUtils::unmergeSubgraph(curNode);
      }
      curNode = nextNode;
    }
    for (Node* n : block_->nodes()) {
      for (Block* b : n->blocks()) {
        ZENDNNSubgraphSlicer(b, graph_, aliasDb_).computeSubgraphsInZENDNN();
      }
    }
  }

  bool shouldConsiderForMerge(Node* node) {
    // if we're already in the process of merging
    if (node->kind() == prim::ZENDNNGroup) {
      return true;
    }
    return frozenZendnnCompatibleLinearNode(node) ||
        frozenZendnnCompatibleConvNode(node) || computableInZENDNN(node);
  }

  std::pair<graph_node_list::iterator, bool> scanNode(Node* producer) {
    if (ZENDNNGroupStart(producer)) {
      if (producer->kind() != prim::ZENDNNGroup) {
        producer = SubgraphUtils::createSingletonSubgraphAndUpdateAliasing(
            producer, prim::ZENDNNGroup, aliasDb_);
      }
      std::vector<Node*> output_nodes;
      for (Value* v : producer->outputs()) {
        for (const auto& use : v->uses()) {
          output_nodes.push_back(use.user);
        }
      }
      std::sort(
          output_nodes.begin(), output_nodes.end(), [&](Node* a, Node* b) {
            return a->isBefore(b);
          });
      for (auto output_node : output_nodes) {
        if (auto group = tryMerge(producer, output_node)) {
          // we successfully merged, so the new group's `outputs` may have
          // changed. So rescan the new group for more merging opportunities.
          return std::make_pair(group.value()->iterator()++, true);
        }
      }
    }

    return std::make_pair(++producer->iterator(), false);
  }

  // Try to merge `consumer` into `producer`. If successful, this destroys
  // `consumer` and returns the `producer` group.
  c10::optional<Node*> tryMerge(Node* producer, Node* consumer) {
    AT_ASSERT(producer->kind() == prim::ZENDNNGroup);
    bool canMerge = shouldConsiderForMerge(consumer) &&
        aliasDb_.moveAfterTopologicallyValid(consumer, producer);

    if (!canMerge) {
      return c10::nullopt;
    }

    SubgraphUtils::mergeNodeIntoSubgraphAndUpdateAliasing(
        consumer, producer, aliasDb_);

    return producer;
  }

  Block* block_;
  std::shared_ptr<Graph> graph_;
  AliasDb& aliasDb_;
};

bool containsZENDNNGroup(Block* b) {
  for (Node* n : b->nodes()) {
    for (Block* block : n->blocks()) {
      if (containsZENDNNGroup(block)) {
        return true;
      }
    }
    if (ZENDNNSubgraphSlicer::ZENDNNGroupStart(n)) {
      return true;
    }
  }
  return false;
}

} // namespace

void ConvertFrozenOpsToZENDNN(std::shared_ptr<Graph>& graph) {
  GRAPH_DUMP("Before convert frozen ops to zendnn", graph);
  // TODO: replace conv1d with conv2d ?
  graph_rewrite_helper::replaceConvolutionWithAtenConv(graph);
  if (containsZENDNNGroup(graph->block())) {
    // Only remove tensor mutation if we know we're going to create speedups
    // with zendnn. Only supporting functional ops simplifies this pass bc
    // running an op in zendnn removes the aliasing relationships that
    // previously existed between input and output.
    RemoveTensorMutation(graph, [](Node* node_to_functionalize) {
      static std::unordered_set<Symbol> zendnn_ops = {
          aten::add_,
          aten::mul_,
          aten::relu_,
          aten::relu6_,
          aten::gelu_,
          aten::hardswish_,
          aten::dropout_,
          aten::sigmoid_,
          aten::hardsigmoid_,
          aten::hardtanh_,
          aten::tanh_,
          aten::clamp_,
      };
      return zendnn_ops.count(node_to_functionalize->kind()) != 0;
    });

    AliasDb db(graph);
    ZENDNNSubgraphSlicer(graph->block(), graph, db).run();
    EliminateDeadCode(graph);
    GRAPH_DUMP("After convert frozen ops to zendnn", graph);
  } else {
    GRAPH_DUMP("No zendnn compatible frozen nodes", graph);
  }
}

#else

void ConvertFrozenOpsToZENDNN(std::shared_ptr<Graph>& graph) {
  GRAPH_DUMP("ZENDNN Not enabled", graph);
}

#endif

} // namespace jit
} // namespace torch
