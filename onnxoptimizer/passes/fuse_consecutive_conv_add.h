#pragma once

#include <cstdio>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/fuse_bn_into_conv.h"

namespace ONNX_NAMESPACE {
namespace optimization {
// TODO: Currently broken for complex values and float16
struct FuseConsecutiveConvAdd final : public PredicateBasedPass {
  explicit FuseConsecutiveConvAdd()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "fuse_consecutive_conv_add";
  }

  bool patternMatchPredicate(Node* node) override {
    // follow the design of fuse_consecutive_concat. Avoid wasting loop here
    return node->kind() == kAdd;
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    Tensor add_tensor;
    Tensor bias_tensor;
    Value* replace_input;
    int num_condition_matched = 0;

    for (size_t i = 0; i < n->inputs().size(); i++) {
      Value* cur_value = n->inputs()[i];
      auto it = graph.getInitializer(cur_value->uniqueName());
      if (it != graph.initializers().end()) {
        add_tensor = *it;
        num_condition_matched++;
      }
      Node* cur_node = cur_value->node();
      if (cur_node->kind() == kConv && cur_value->uses().size() == 1) {
        const auto conv_input = cur_node->inputs();
        auto b_iter = graph.getInitializer(conv_input[2]->uniqueName());
        bias_tensor = *b_iter;
        replace_input = cur_value;
        num_condition_matched++;
      }
    }

    if (num_condition_matched == 2) {
#define DO_COMPUTATION(t1, t2)                          \
  t2 add_constant = add_tensor.data<t2>()[0];           \
  for (size_t i = 0; i < bias_tensor.sizes()[0]; i++) { \
    bias_tensor.data<t1>()[i] += add_constant;          \
  }
#define DO_COMPUTATION_WITH_BIAS_KNOWN(t1)              \
  switch (add_tensor.elem_type()) {                     \
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {  \
      DO_COMPUTATION(t1, float)                         \
      break;                                            \
    }                                                   \
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: { \
      DO_COMPUTATION(t1, double)                        \
      break;                                            \
    }                                                   \
    default:                                            \
      return false;                                     \
  }

      switch (bias_tensor.elem_type()) {
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
          DO_COMPUTATION_WITH_BIAS_KNOWN(float)
          break;
        }
        case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
          DO_COMPUTATION_WITH_BIAS_KNOWN(double)
          break;
        }
        default:
          return false;
      }

      Value* new_bias = graph.addInitializerAndInput(bias_tensor);
      Value* old_bias = replace_input->node()->inputs()[2];
      replace_input->node()->replaceInput(2, new_bias);
      if (old_bias->uses().size() == 0) {
        graph.eraseInitializerAndInput(old_bias);
      }
      tryReplacingAllUsesWith(n->output(), replace_input);

      return true;
#undef DO_COMPUTATION_WITH_BIAS_KNOWN
#undef DO_COMPUTATION
    }
    return false;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
