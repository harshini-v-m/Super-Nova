//===- ErrorHandling.h - Async Error Propagation -------------------*- C++ -*-===//
//
// Nova Compiler Runtime
//
//===----------------------------------------------------------------------===//

#ifndef NOVA_RUNTIME_CORE_ERRORHANDLING_H
#define NOVA_RUNTIME_CORE_ERRORHANDLING_H

#include "Runtime/Core/AsyncValue.h"
#include <vector>

namespace nova {
namespace runtime {

class ErrorHandler {
public:
  // Disallow instantiation
  ErrorHandler() = delete;

  // If 'source' has an error, propagate it to all 'targets'.
  // Often used in 'AndThen' callbacks.
  static void PropagateError(AsyncValue *source,
                             const std::vector<AsyncValue *> &targets);

  // Mark downstream values as Error/Cancelled if the source failed.
  static void CancelDownstream(AsyncValue *failed_value);
};

} // namespace runtime
} // namespace nova

#endif // NOVA_RUNTIME_CORE_ERRORHANDLING_H
