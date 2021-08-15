#pragma once

#include <cstdio>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/utils/log_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
// TODO: Currently broken for complex values and float16
struct WarnResize final : public PredicateBasedPass {
  explicit WarnResize()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "warn_resize";
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
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE