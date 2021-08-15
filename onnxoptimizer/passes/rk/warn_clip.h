#pragma once

#include <cstdio>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/utils/log_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
// TODO: Currently broken for complex values and float16
struct WarnClip final : public PredicateBasedPass {
  explicit WarnClip()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "warn_clip";
  }

  bool patternMatchPredicate(Node* node) override {
    return std::string(node->kind().toString()).compare("Clip") == 0;
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    const auto node_name = n->name();
    LOGW("[NODE: %s] quantized clip op's precision on RK npu is unstable\n",
         node_name.c_str());
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE