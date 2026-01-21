//===- ExecutionPlan.h - Runtime Execution Plan --------------------*- C++ -*-===//
//
// Nova Compiler Runtime
//
//===----------------------------------------------------------------------===//

#ifndef NOVA_RUNTIME_EXECUTOR_EXECUTIONPLAN_H
#define NOVA_RUNTIME_EXECUTOR_EXECUTIONPLAN_H

#include "Runtime/Executor/KernelRegistry.h"
#include "Runtime/Executor/DependencyGraph.h"
#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace nova {
namespace runtime {

// Forward declarations
// Forward declarations
// class DependencyGraph;

// Tensor metadata (shape, dtype, device)
struct TensorMetadata {
  std::vector<int64_t> shape;
  int dtype;  // Maps to OwnTensor::Dtype
  int device; // Maps to OwnTensor::DeviceIndex
};

// Task argument types
struct ArgInput {
  int idx; // Index into inputs array
};

struct ArgParam {
  int idx; // Index into params array
};

struct ArgSlot {
  int slot; // Index into intermediate slots
};

struct ArgLiteral {
  // For now, just store as a generic value
  // In real implementation, this would be a Tensor
  void* data = nullptr;
};

using TaskArg = std::variant<ArgInput, ArgParam, ArgSlot, ArgLiteral>;

// Async task structure
struct AsyncTask {
  int task_id;
  std::string op_name;
  Device device;
  
  // Dependencies (task IDs that must complete first)
  std::vector<int> dependencies;
  
  // Arguments
  std::vector<TaskArg> args;
  
  // Output metadata
  TensorMetadata output_meta;
  
  // JIT artifacts (optional)
  void* jit_function = nullptr;  // CPU: function pointer
  void* cuda_module = nullptr;   // GPU: CUmodule
};

// Runtime execution plan
struct RuntimeExecutionPlan {
  std::vector<AsyncTask> tasks;
  int output_task_id;
  
  // Input/param metadata
  std::vector<TensorMetadata> input_meta;
  std::vector<TensorMetadata> param_meta;
  
  // Cached dependency graph
  std::unique_ptr<DependencyGraph> dep_graph;
};

} // namespace runtime
} // namespace nova

#endif // NOVA_RUNTIME_EXECUTOR_EXECUTIONPLAN_H
