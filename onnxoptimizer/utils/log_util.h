#pragma once

#include <cstdlib>

#define __FILENAME__ \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define LOGI(...)          \
  ::std::printf("[INFO]"); \
  ::std::printf(__VA_ARGS__)

#define LOGD(...)                                   \
  ::std::printf("[DEBUG]");                         \
  ::std::printf("[%s:%d]", __FILENAME__, __LINE__); \
  ::std::printf(__VA_ARGS__)

#define LOGE(...)                 \
  ::fprintf(stderr, "\x1b[31m");  \
  ::fprintf(stderr, "[ERROR]");   \
  ::fprintf(stderr, __VA_ARGS__); \
  ::fprintf(stderr, "\x1b[0m")

#define LOGW(...)             \
  ::std::printf("\x1b[33m");  \
  ::std::printf("[WARNING]"); \
  ::std::printf(__VA_ARGS__); \
  ::std::printf("\x1b[0m")

namespace ONNX_NAMESPACE {
namespace optimization {}  // namespace optimization
}  // namespace ONNX_NAMESPACE
