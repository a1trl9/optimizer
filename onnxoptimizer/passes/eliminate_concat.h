#pragma once

#include <cstdio>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {
// TODO: Currently broken for complex values and float16
struct EliminateConcat final : public PredicateBasedPass {
  explicit EliminateConcat()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_concat";
  }

  bool patternMatchPredicate(Node* node) override {
    return node->kind() == kConcat && node->inputs().size() == 1;
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    auto input = n->inputs()[0];
    tryReplacingAllUsesWith(n->output(), input);
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE