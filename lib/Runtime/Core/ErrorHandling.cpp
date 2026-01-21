//===- ErrorHandling.cpp - Async Error Propagation ------------------------===//
//
// Nova Compiler Runtime
//
//===----------------------------------------------------------------------===//

#include "Runtime/Core/ErrorHandling.h"
#include <iostream>

namespace nova {
namespace runtime {

void ErrorHandler::PropagateError(AsyncValue *source,
                                  const std::vector<AsyncValue *> &targets) {
  if (!source->IsError())
    return;

  std::string msg = source->GetError();
  for (auto *target : targets) {
    if (target->IsUnconstructed()) {
      target->SetError(msg);
    }
  }
}

void ErrorHandler::CancelDownstream(AsyncValue *failed_value) {
  // Simple implementation: just log for now, or if we had a dependency graph, walk it.
  // In TFRT, this might involve checking a CancellationContext.
  // For Phase 1 scaffold:
  if (failed_value->IsError()) {
    // std::cerr << "Cancelling downstream due to error: " << failed_value->GetError() << "\n";
  }
}

} // namespace runtime
} // namespace nova
