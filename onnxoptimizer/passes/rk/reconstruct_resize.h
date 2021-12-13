#pragma once

#include <cstdio>
#include <cmath>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/utils/log_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
// TODO: Currently broken for complex values and float16
struct ReconstructResize final : public PredicateBasedPass {
  explicit ReconstructResize()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "reconstruct_resize";
  }

  bool patternMatchPredicate(Node* node) override {
    return std::string(node->kind().toString()).compare("Resize") == 0;
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    const auto node_name = n->name();
    LOGW(
        "[NODE: %s] resize op's behavior on RK npu is different from "
        "PyTorch\n",
        node_name.c_str());

    if (n->inputs().size() != 4) {
      return true;
    }

    // RK does not support Resize Operator when `size` attribute is used
    // Also output shape should be determined

    Tensor sizes_tensor;
    Tensor scales_tensor;
    Tensor new_sizes_tensor;

    auto x_input = n->inputs()[0];
    auto scales_input = n->inputs()[2];
    auto sizes_input = n->inputs()[3];

    auto x_sizes_vec = x_input->sizes();
    int64_t input_height = x_sizes_vec[2].dim;
    int64_t input_width = x_sizes_vec[3].dim;

    auto sizes_it = graph.getInitializer(sizes_input->uniqueName());
    if (sizes_it != graph.initializers().end()) {
      sizes_tensor = *sizes_it;
    } else {
      return true;
    }

    auto size_vec = sizes_tensor.data<int64_t>();
    int64_t height = size_vec[2];
    int64_t width = size_vec[3];

    float scale_h = static_cast<float>(height) / static_cast<float>(input_height);
    float scale_w = static_cast<float>(width) / static_cast<float>(input_width);

    // RK's shape inference: ceiling
    if (ceil(scale_h) != scale_h) {
      scale_h = static_cast<float>(height + 0.4) / static_cast<float>(input_height);
    }

    if (ceil(scale_w) != scale_w) {
      scale_w = static_cast<float>(width + 0.4) / static_cast<float>(input_width);
    }

    // update output
    auto y_output = n->outputs()[0];
    std::vector<Dimension> d_sizes{Dimension(1),
                                   Dimension(x_input->sizes()[1].dim),
                                   Dimension(height), Dimension(width)};
    y_output->setSizes(d_sizes);

    // update scales, so far only support float
    scales_tensor.sizes().push_back(4);
    scales_tensor.elem_type() = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
    scales_tensor.floats().push_back(1);
    scales_tensor.floats().push_back(1);
    scales_tensor.floats().push_back(scale_h);
    scales_tensor.floats().push_back(scale_w);

    Value* new_scales_input = graph.addInitializerAndInput(scales_tensor);
    n->replaceInput(2, new_scales_input);

    // remove old size and scale inputs/initializers if possible
    n->removeInput(3);
    if (scales_input->uses().size() == 0) {
      graph.eraseInitializerAndInput(scales_input);
    }

    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
