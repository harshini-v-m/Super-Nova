//===- ExecutionEngine.cpp - Async Graph Executor -------------------------===//
//
// Nova Compiler Runtime
//
//===----------------------------------------------------------------------===//

#include "Runtime/Executor/ExecutionEngine.h"
#include "Runtime/Executor/DependencyGraph.h"
#include "Runtime/Core/ErrorHandling.h"
#include <iostream>
#include <stdexcept>

namespace nova {
namespace runtime {

ExecutionEngine::ExecutionEngine(HostContext* host)
    : host_(host), registry_(&KernelRegistry::Instance()) {}

ExecutionEngine::~ExecutionEngine() = default;

AsyncValue* ExecutionEngine::Execute(
    const RuntimeExecutionPlan& plan,
    const std::vector<void*>& inputs,
    const std::vector<void*>& params) {
  
  // Create intermediate AsyncValues for each task
  std::vector<AsyncValue*> intermediate_values(plan.tasks.size(), nullptr);
  
  // Create dependency graph if not cached
  DependencyGraph dep_graph;
  for (const auto& task : plan.tasks) {
    dep_graph.AddTask(task.task_id, task.dependencies);
  }
  
  // Dispatch all tasks
  for (const auto& task : plan.tasks) {
    // Create AsyncValue for this task's output
    intermediate_values[task.task_id] = host_->MakeUnconstructedAsyncValue<void*>();
    
    // Get dependencies
    std::vector<AsyncValue*> deps;
    for (int dep_id : task.dependencies) {
      deps.push_back(intermediate_values[dep_id]);
    }
    
    // Dispatch task when dependencies are ready
    if (deps.empty()) {
      // No dependencies, dispatch immediately
      DispatchTask(task, intermediate_values, inputs, params);
    } else {
      // Wait for dependencies
      // CRITICAL FIX: Capture intermediate_values BY VALUE (copy vector of pointers)
      // because local vector dies when Execute returns.
      RunWhenReady(deps, [this, &task, intermediate_values, &inputs, &params]() {
        DispatchTask(task, intermediate_values, inputs, params);
      });
    }
  }
  
  // Return the output AsyncValue
  return intermediate_values[plan.output_task_id];
}

void* ExecutionEngine::ExecuteSync(
    const RuntimeExecutionPlan& plan,
    const std::vector<void*>& inputs,
    const std::vector<void*>& params) {
  
  auto* result = Execute(plan, inputs, params);
  result->Await();
  
  if (result->IsError()) {
    throw std::runtime_error("Execution failed: " + result->GetError());
  }
  
  // In real implementation, this would return Tensor*
  // For now, return nullptr as placeholder
  return nullptr;
}

void ExecutionEngine::DispatchTask(
    const AsyncTask& task,
    const std::vector<AsyncValue*>& intermediate_values,
    const std::vector<void*>& inputs,
    const std::vector<void*>& params) {
  
  // Get the output AsyncValue for this task
  AsyncValue* output = intermediate_values[task.task_id];
  
  // Enqueue work on thread pool
  host_->EnqueueWork([this, &task, output, intermediate_values, &inputs, &params]() {
    try {
      // Resolve arguments for BOTH paths
      std::vector<AsyncValue*> kernel_args;
      for (const auto& arg : task.args) {
        AsyncValue* resolved = ResolveArg(arg, intermediate_values, inputs, params);
        kernel_args.push_back(resolved);
      }

      AsyncValue* result = nullptr;

      if (task.jit_function) {
         // --- JIT Path ---
         // Signature: void* func(void** args)
         using JITFunc = void* (*)(void**);
         JITFunc func = reinterpret_cast<JITFunc>(task.jit_function);

         // 1. Unwrap AsyncValues to raw pointers
         std::vector<void*> raw_args;
         raw_args.reserve(kernel_args.size());
         for(auto* av : kernel_args) {
            if(auto* concrete = dynamic_cast<ConcreteAsyncValue<void*>*>(av)) {
                raw_args.push_back(concrete->get());
            } else {
                raw_args.push_back(nullptr); // Should handle error better
            }
         }

         // 2. Call the compiled function
         // We expect the JIT function to return the result pointer (e.g. Tensor*)
         void* raw_result = func(raw_args.data());
         
         // 3. Wrap result in AsyncValue
         result = host_->MakeAvailableAsyncValue<void*>(raw_result);

      } else {
         // --- Library Path ---
         // Get the kernel for this operation from Registry
         KernelFunction kernel = registry_->GetKernel(task.op_name, task.device);
    
         // Execute kernel wrapper
         result = kernel(kernel_args, host_);
      }
      
      // For now, just mark output as complete
      // In real implementation, we'd transfer the result
      // Transfer result from kernel to the output AsyncValue
      // We assume kernel returns AsyncValue<void*> as per convention for our runtime
      if (auto* concrete_result = dynamic_cast<ConcreteAsyncValue<void*>*>(result)) {
         if (auto* concrete_output = dynamic_cast<ConcreteAsyncValue<void*>*>(output)) {
           // We propagate the pointer (Tensor*)
           concrete_output->emplace(concrete_result->get());
         }
      } else {
        // Fallback for void results
        if (auto* concrete_output = dynamic_cast<ConcreteAsyncValue<void>*>(output)) {
             concrete_output->emplace();
        }
      }
      
    } catch (const std::exception& e) {
      // Set error on output
      output->SetError(e.what());
    }
  });
}

void ExecutionEngine::RunWhenReady(
    const std::vector<AsyncValue*>& deps,
    std::function<void()> callback) {
  
  if (deps.empty()) {
    callback();
    return;
  }
  
  // Count remaining dependencies
  auto remaining = std::make_shared<std::atomic<int>>(deps.size());
  
  for (auto* dep : deps) {
    dep->AndThen([remaining, callback](AsyncValue* av) {
      if (av->IsError()) {
        // Propagate error - don't execute callback
        return;
      }
      
      int count = remaining->fetch_sub(1, std::memory_order_acq_rel);
      if (count == 1) {
        // All dependencies ready
        callback();
      }
    });
  }
}

AsyncValue* ExecutionEngine::ResolveArg(
    const TaskArg& arg,
    const std::vector<AsyncValue*>& intermediate_values,
    const std::vector<void*>& inputs,
    const std::vector<void*>& params) {
  
  return std::visit([&](auto&& a) -> AsyncValue* {
    using T = std::decay_t<decltype(a)>;
    
    if constexpr (std::is_same_v<T, ArgInput>) {
      // Wrap input tensor in an available AsyncValue
      // In real implementation: return host_->MakeAvailableAsyncValue<Tensor>(*inputs[a.idx]);
      return host_->MakeAvailableAsyncValue<void*>(inputs[a.idx]);
      
    } else if constexpr (std::is_same_v<T, ArgParam>) {
      // Wrap param tensor in an available AsyncValue
      return host_->MakeAvailableAsyncValue<void*>(params[a.idx]);
      
    } else if constexpr (std::is_same_v<T, ArgSlot>) {
      // Return intermediate value
      return intermediate_values[a.slot];
      
    } else if constexpr (std::is_same_v<T, ArgLiteral>) {
      // Wrap literal in an available AsyncValue
      // Note: Passing nullptr for now as placeholder for literal data
      return host_->MakeAvailableAsyncValue<void*>(nullptr);
    }
    
    return nullptr;
  }, arg);
}

} // namespace runtime
} // namespace nova
