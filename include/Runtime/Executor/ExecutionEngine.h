//===- ExecutionEngine.h - Async Graph Executor --------------------*- C++ -*-===//
//
// Nova Compiler Runtime
//
//===----------------------------------------------------------------------===//

#ifndef NOVA_RUNTIME_EXECUTOR_EXECUTIONENGINE_H
#define NOVA_RUNTIME_EXECUTOR_EXECUTIONENGINE_H

#include "Runtime/Core/AsyncValue.h"
#include "Runtime/Core/HostContext.h"
#include "Runtime/Executor/ExecutionPlan.h"
#include "Runtime/Executor/KernelRegistry.h"
#include <memory>
#include <vector>

namespace nova {
namespace runtime {

// Forward declaration - will be replaced with actual Tensor type
class TensorPlaceholder;

class ExecutionEngine {
public:
  explicit ExecutionEngine(HostContext* host);
  ~ExecutionEngine();
  
  // Execute a compiled plan asynchronously
  // Returns AsyncValue that will contain the output tensor
  AsyncValue* Execute(
      const RuntimeExecutionPlan& plan,
      const std::vector<void*>& inputs,  // Tensor* in real implementation
      const std::vector<void*>& params); // Tensor* in real implementation
  
  // Synchronous execute (blocks until complete)
  // Returns the output tensor directly
  void* ExecuteSync(
      const RuntimeExecutionPlan& plan,
      const std::vector<void*>& inputs,
      const std::vector<void*>& params);

private:
  // Dispatch a single task
  void DispatchTask(const AsyncTask& task,
                    const std::vector<AsyncValue*>& intermediate_values,
                    const std::vector<void*>& inputs,
                    const std::vector<void*>& params);
  
  // Helper: Run callback when all dependencies are ready
  void RunWhenReady(const std::vector<AsyncValue*>& deps,
                    std::function<void()> callback);
  
  // Helper: Resolve task argument to AsyncValue
  AsyncValue* ResolveArg(const TaskArg& arg,
                         const std::vector<AsyncValue*>& intermediate_values,
                         const std::vector<void*>& inputs,
                         const std::vector<void*>& params);
  
  HostContext* host_;
  KernelRegistry* registry_;
};

} // namespace runtime
} // namespace nova

#endif // NOVA_RUNTIME_EXECUTOR_EXECUTIONENGINE_H
