# Owner(s): ["oncall: fx"]

import os
import pickle
import importlib
import logging

import torch
from torch.fx._symbolic_trace import symbolic_trace

from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.backends.nvfuser.operator_support import NvFuserOperatorSupport
import torch._prims as prims
from torch.fx.passes.graph_drawer import FxGraphDrawer
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS

from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from torch.testing._internal.jit_utils import JitTestCase

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestFXGraphPasses(JitTestCase):

    def test_nvfuser_patition_real_models(self):
        if not os.path.exists("torch_bench_graphs"):
            self.skipTest("torch_bench_graphs doesn't exist, skipping tests....")

        # this test assumes torch_bench_graphs is place at torch's root folder
        test_cases = [
            "torch_bench_graphs/resnext50_32x4d/resnext50_32x4d_forward_0",
            "torch_bench_graphs/resnext50_32x4d/resnext50_32x4d_backward_0",
            "torch_bench_graphs/nvidia_deeprecommender/nvidia_deeprecommender_backward_0",
            "torch_bench_graphs/nvidia_deeprecommender/nvidia_deeprecommender_forward_0",
            "torch_bench_graphs/moco/moco_forward_4",
            "torch_bench_graphs/moco/moco_backward_0",
            "torch_bench_graphs/moco/moco_backward_7",
            "torch_bench_graphs/moco/moco_forward_9",
            "torch_bench_graphs/moco/moco_forward_3",
            "torch_bench_graphs/moco/moco_backward_10",
            "torch_bench_graphs/moco/moco_forward_7",
            "torch_bench_graphs/moco/moco_backward_9",
            "torch_bench_graphs/moco/moco_backward_3",
            "torch_bench_graphs/moco/moco_forward_10",
            "torch_bench_graphs/moco/moco_backward_4",
            "torch_bench_graphs/moco/moco_forward_0",
            "torch_bench_graphs/moco/moco_backward_6",
            "torch_bench_graphs/moco/moco_forward_5",
            "torch_bench_graphs/moco/moco_backward_2",
            "torch_bench_graphs/moco/moco_forward_2",
            "torch_bench_graphs/moco/moco_forward_8",
            "torch_bench_graphs/moco/moco_backward_11",
            "torch_bench_graphs/moco/moco_backward_1",
            "torch_bench_graphs/moco/moco_backward_5",
            "torch_bench_graphs/moco/moco_forward_1",
            "torch_bench_graphs/moco/moco_forward_6",
            "torch_bench_graphs/moco/moco_backward_8",
            "torch_bench_graphs/moco/moco_forward_11",
            "torch_bench_graphs/resnet18/resnet18_backward_0",
            "torch_bench_graphs/resnet18/resnet18_forward_0",
            "torch_bench_graphs/mnasnet1_0/mnasnet1_0_backward_0",
            "torch_bench_graphs/mnasnet1_0/mnasnet1_0_forward_0",
            "torch_bench_graphs/BERT_pytorch/BERT_pytorch_forward_0",
            "torch_bench_graphs/BERT_pytorch/BERT_pytorch_backward_0",
            "torch_bench_graphs/resnet50/resnet50_forward_0",
            "torch_bench_graphs/resnet50/resnet50_backward_0",
            "torch_bench_graphs/hf_DistilBert/hf_DistilBert_backward_0",
            "torch_bench_graphs/hf_DistilBert/hf_DistilBert_forward_1",
            "torch_bench_graphs/hf_DistilBert/hf_DistilBert_forward_0",
            "torch_bench_graphs/hf_DistilBert/hf_DistilBert_backward_1",
            "torch_bench_graphs/hf_Albert/hf_Albert_backward_1",
            "torch_bench_graphs/hf_Albert/hf_Albert_forward_3",
            "torch_bench_graphs/hf_Albert/hf_Albert_backward_2",
            "torch_bench_graphs/hf_Albert/hf_Albert_forward_0",
            "torch_bench_graphs/hf_Albert/hf_Albert_forward_2",
            "torch_bench_graphs/hf_Albert/hf_Albert_backward_0",
            "torch_bench_graphs/hf_Albert/hf_Albert_forward_1",
            "torch_bench_graphs/hf_Albert/hf_Albert_backward_3",
            "torch_bench_graphs/dlrm/dlrm_backward_0",
            "torch_bench_graphs/dlrm/dlrm_forward_0",
            "torch_bench_graphs/drq/drq_backward_0",
            "torch_bench_graphs/drq/drq_forward_1",
            "torch_bench_graphs/drq/drq_backward_1",
            "torch_bench_graphs/drq/drq_forward_0",
            "torch_bench_graphs/pytorch_struct/pytorch_struct_backward_0",
            "torch_bench_graphs/pytorch_struct/pytorch_struct_forward_0",
            "torch_bench_graphs/Background_Matting/Background_Matting_backward_0",
            "torch_bench_graphs/Background_Matting/Background_Matting_forward_0",
            "torch_bench_graphs/timm_regnet/timm_regnet_forward_0",
            "torch_bench_graphs/timm_regnet/timm_regnet_backward_0",
            "torch_bench_graphs/hf_Bert/hf_Bert_forward_1",
            "torch_bench_graphs/hf_Bert/hf_Bert_backward_1",
            "torch_bench_graphs/hf_Bert/hf_Bert_backward_2",
            "torch_bench_graphs/hf_Bert/hf_Bert_forward_2",
            "torch_bench_graphs/hf_Bert/hf_Bert_forward_0",
            "torch_bench_graphs/hf_Bert/hf_Bert_backward_0",
            "torch_bench_graphs/densenet121/densenet121_backward_0",
            "torch_bench_graphs/densenet121/densenet121_forward_0",
            "torch_bench_graphs/timm_nfnet/timm_nfnet_backward_0",
            "torch_bench_graphs/timm_nfnet/timm_nfnet_forward_0",
            "torch_bench_graphs/squeezenet1_1/squeezenet1_1_forward_0",
            "torch_bench_graphs/squeezenet1_1/squeezenet1_1_backward_0",
            "torch_bench_graphs/alexnet/alexnet_forward_0",
            "torch_bench_graphs/alexnet/alexnet_backward_0",
            "torch_bench_graphs/Super_SloMo/Super_SloMo_forward_0",
            "torch_bench_graphs/Super_SloMo/Super_SloMo_backward_0",
            "torch_bench_graphs/timm_vision_transformer/timm_vision_transformer_backward_0",
            "torch_bench_graphs/timm_vision_transformer/timm_vision_transformer_forward_0",
            "torch_bench_graphs/maml_omniglot/maml_omniglot_backward_0",
            "torch_bench_graphs/maml_omniglot/maml_omniglot_forward_0",
            "torch_bench_graphs/hf_Bart/hf_Bart_forward_1",
            "torch_bench_graphs/hf_Bart/hf_Bart_forward_13",
            "torch_bench_graphs/hf_Bart/hf_Bart_backward_0",
            "torch_bench_graphs/hf_Bart/hf_Bart_backward_7",
            "torch_bench_graphs/hf_Bart/hf_Bart_forward_6",
            "torch_bench_graphs/hf_Bart/hf_Bart_backward_11",
            "torch_bench_graphs/hf_Bart/hf_Bart_backward_9",
            "torch_bench_graphs/hf_Bart/hf_Bart_backward_3",
            "torch_bench_graphs/hf_Bart/hf_Bart_forward_10",
            "torch_bench_graphs/hf_Bart/hf_Bart_forward_2",
            "torch_bench_graphs/hf_Bart/hf_Bart_forward_8",
            "torch_bench_graphs/hf_Bart/hf_Bart_backward_12",
            "torch_bench_graphs/hf_Bart/hf_Bart_forward_5",
            "torch_bench_graphs/hf_Bart/hf_Bart_backward_4",
            "torch_bench_graphs/hf_Bart/hf_Bart_backward_6",
            "torch_bench_graphs/hf_Bart/hf_Bart_backward_10",
            "torch_bench_graphs/hf_Bart/hf_Bart_forward_7",
            "torch_bench_graphs/hf_Bart/hf_Bart_forward_12",
            "torch_bench_graphs/hf_Bart/hf_Bart_forward_0",
            "torch_bench_graphs/hf_Bart/hf_Bart_backward_1",
            "torch_bench_graphs/hf_Bart/hf_Bart_forward_4",
            "torch_bench_graphs/hf_Bart/hf_Bart_backward_13",
            "torch_bench_graphs/hf_Bart/hf_Bart_backward_5",
            "torch_bench_graphs/hf_Bart/hf_Bart_backward_2",
            "torch_bench_graphs/hf_Bart/hf_Bart_backward_8",
            "torch_bench_graphs/hf_Bart/hf_Bart_forward_9",
            "torch_bench_graphs/hf_Bart/hf_Bart_forward_3",
            "torch_bench_graphs/hf_Bart/hf_Bart_forward_11",
            "torch_bench_graphs/timm_resnest/timm_resnest_forward_0",
            "torch_bench_graphs/timm_resnest/timm_resnest_backward_0",
            "torch_bench_graphs/mobilenet_v2/mobilenet_v2_backward_0",
            "torch_bench_graphs/mobilenet_v2/mobilenet_v2_forward_0",
            "torch_bench_graphs/timm_efficientnet/timm_efficientnet_forward_0",
            "torch_bench_graphs/timm_efficientnet/timm_efficientnet_backward_0",
            "torch_bench_graphs/soft_actor_critic/soft_actor_critic_backward_1",
            "torch_bench_graphs/soft_actor_critic/soft_actor_critic_forward_1",
            "torch_bench_graphs/soft_actor_critic/soft_actor_critic_backward_0",
            "torch_bench_graphs/soft_actor_critic/soft_actor_critic_forward_0",
            "torch_bench_graphs/mobilenet_v2_quantized_qat/mobilenet_v2_quantized_qat_backward_0",
            "torch_bench_graphs/mobilenet_v2_quantized_qat/mobilenet_v2_quantized_qat_forward_0",
            "torch_bench_graphs/LearningToPaint/LearningToPaint_backward_0",
            "torch_bench_graphs/LearningToPaint/LearningToPaint_forward_0",
            "torch_bench_graphs/vgg16/vgg16_forward_0",
            "torch_bench_graphs/vgg16/vgg16_backward_0",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_backward_1",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_forward_6",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_forward_1",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_backward_6",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_forward_11",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_backward_8",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_backward_2",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_forward_5",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_forward_8",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_forward_2",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_backward_5",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_backward_10",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_backward_7",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_forward_0",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_forward_7",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_backward_0",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_backward_4",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_backward_11",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_forward_3",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_forward_9",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_forward_4",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_forward_10",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_backward_3",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_backward_9",
            "torch_bench_graphs/pytorch_unet/pytorch_unet_backward_0",
            "torch_bench_graphs/pytorch_unet/pytorch_unet_forward_0",
            "torch_bench_graphs/dcgan/dcgan_backward_0",
            "torch_bench_graphs/dcgan/dcgan_forward_0",
            "torch_bench_graphs/timm_vovnet/timm_vovnet_forward_0",
            "torch_bench_graphs/timm_vovnet/timm_vovnet_backward_0",
            "torch_bench_graphs/hf_T5/hf_T5_forward_7",
            "torch_bench_graphs/hf_T5/hf_T5_forward_13",
            "torch_bench_graphs/hf_T5/hf_T5_backward_0",
            "torch_bench_graphs/hf_T5/hf_T5_backward_11",
            "torch_bench_graphs/hf_T5/hf_T5_backward_7",
            "torch_bench_graphs/hf_T5/hf_T5_forward_0",
            "torch_bench_graphs/hf_T5/hf_T5_forward_14",
            "torch_bench_graphs/hf_T5/hf_T5_backward_9",
            "torch_bench_graphs/hf_T5/hf_T5_backward_3",
            "torch_bench_graphs/hf_T5/hf_T5_forward_10",
            "torch_bench_graphs/hf_T5/hf_T5_forward_4",
            "torch_bench_graphs/hf_T5/hf_T5_backward_12",
            "torch_bench_graphs/hf_T5/hf_T5_forward_9",
            "torch_bench_graphs/hf_T5/hf_T5_forward_3",
            "torch_bench_graphs/hf_T5/hf_T5_backward_4",
            "torch_bench_graphs/hf_T5/hf_T5_backward_6",
            "torch_bench_graphs/hf_T5/hf_T5_forward_1",
            "torch_bench_graphs/hf_T5/hf_T5_backward_10",
            "torch_bench_graphs/hf_T5/hf_T5_forward_12",
            "torch_bench_graphs/hf_T5/hf_T5_forward_6",
            "torch_bench_graphs/hf_T5/hf_T5_backward_1",
            "torch_bench_graphs/hf_T5/hf_T5_forward_2",
            "torch_bench_graphs/hf_T5/hf_T5_forward_8",
            "torch_bench_graphs/hf_T5/hf_T5_backward_5",
            "torch_bench_graphs/hf_T5/hf_T5_backward_13",
            "torch_bench_graphs/hf_T5/hf_T5_backward_14",
            "torch_bench_graphs/hf_T5/hf_T5_backward_2",
            "torch_bench_graphs/hf_T5/hf_T5_backward_8",
            "torch_bench_graphs/hf_T5/hf_T5_forward_5",
            "torch_bench_graphs/hf_T5/hf_T5_forward_11",
            "torch_bench_graphs/shufflenet_v2_x1_0/shufflenet_v2_x1_0_backward_0",
            "torch_bench_graphs/shufflenet_v2_x1_0/shufflenet_v2_x1_0_forward_0",
        ]

        device = 'cuda'
        draw = False

        for dir in test_cases:
            path = dir.split('/')
            model_name = path[-1]
            module_path = '.'.join(path)
            input_data_path = f'{dir}/{model_name}.input'

            print(f"====== {model_name} ======")

            module = importlib.import_module(module_path)

            try:
                m = module.FxModule()
                m.to(device)

                traced = symbolic_trace(m)

                if draw:
                    print("Drawing original graph...")
                    drawer = FxGraphDrawer(traced, "test")
                    dot_graph = drawer.get_dot_graph()
                    dot_graph.write_png("before.png")

                supported_ops = NvFuserOperatorSupport()
                partitioner = CapabilityBasedPartitioner(traced, supported_ops)

                fused_graph_module = partitioner.partition_and_fuse()

                # compile the nvFuser submodel with torchscript jit
                # for node in fused_graph.graph.nodes:
                #     if "fused_" in node.name:
                #         module = getattr(fused_graph, node.name)
                #         setattr(fused_graph, node.name, torch.jit.script(module) )


                # for node in fused_graph.graph.nodes:
                #     if "fused_" in node.name:
                        # module = getattr(fused_graph, node.name)
                        # setattr(fused_graph, node.name, torch.jit.script(module) )


                # if draw:
                #     print("Drawing fused graph...")
                #     drawer = FxGraphDrawer(fused_graph_module, "test")
                #     dot_graph = drawer.get_dot_graph()
                #     dot_graph.write_png("after.png")

                print("Generating testing data...")
                with (open(input_data_path, 'rb')) as f:
                    inputs_meta = pickle.load(f)

                    inputs = []
                    for meta in inputs_meta:
                        type, shape, stride, dtype = meta

                        if dtype in {torch.int, torch.int32, torch.int64, torch.bool, torch.int, torch.uint8}:
                            input = torch.randint(0, 1, shape, dtype=dtype, device=device)
                        else:
                            input = torch.rand(shape, dtype=dtype, device=device)

                        inputs.append(input)

                # m.to(device)
                # fused_graph_module.to(device)

                print("Running original model...")
                expected = m(*inputs)

                print("Running fused model...")
                result = fused_graph_module(*inputs)





                # def compiler_fn(fx_module: torch.fx.GraphModule, args):
                #     new_graph = torch.fx.Graph()
                #     decomp = dict(decomposition_table)
                #     decomp[torch.ops.aten.to.dtype] = aten_to_dtype
                #     DecompositionInterpreter(fx_module, new_graph, decomposition_table=decomp).run(*args)

                #     res = torch.fx.GraphModule(fx_module, new_graph)
                #     print(res.code)

                #     def with_nvfuser(*args):
                #         return execute(res, *args, executor="nvfuser")

                #     return with_nvfuser



                torch.testing.assert_close(expected, result, equal_nan=True, rtol=1e-5, atol=1e-5)
                print(f"{model_name} Passed!")

            except Exception as e:
                print(f"{model_name} failed!", e)


    # def test_nvfuser_prim_operator_support(self):
    #     def _wrapper(a, b, broadcast_dimensions):
    #         a_bc = prims.broadcast_in_dim(a, b.shape, broadcast_dimensions)
    #         return prims.add(a_bc, b)

    #     traced = symbolic_trace(_wrapper)

    #     supported_ops = NvFuserOperatorSupport()
    #     for node in traced.graph.nodes:
    #         if node.op in CALLABLE_NODE_OPS:
    #             assert supported_ops.is_node_supported({}, node)



instantiate_parametrized_tests(TestFXGraphPasses)

if __name__ == "__main__":
    run_tests()
