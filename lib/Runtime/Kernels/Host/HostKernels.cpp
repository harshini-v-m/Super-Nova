//===- HostKernels.cpp - CPU Kernel Wrappers ------------------------------===//
//
// Nova Compiler Runtime
//
//===----------------------------------------------------------------------===//

#include "Runtime/Kernels/KernelRegistration.h"
#include "Runtime/Core/AsyncValue.h"
#include "Runtime/Core/HostContext.h"
#include <iostream>

// --- TensorLib Integration ---
// Since TensorLib might be linked externally, we assume headers are available.
// If not available during this build step, we'll mock the operations for the scaffold.
#if __has_include("core/Tensor.h") || defined(HAS_TENSORLIB)
#include "core/Tensor.h"
#include "ops/TensorOps.h"
using Tensor = OwnTensor::Tensor;
#else
// Mock Tensor for standalone compilation if headers missing
namespace OwnTensor {
  struct Tensor { 
    // minimal mock
    static Tensor zeros(std::vector<long>, bool) { return {}; }
  }; 
}
using Tensor = OwnTensor::Tensor;
#endif

namespace nova {
namespace runtime {

// --- Helper: AsyncValue Unwrap/Wrap ---

// Example Wrapper for "nova.add"
// Inputs: [AsyncValue<Tensor>, AsyncValue<Tensor>]
// Output: [AsyncValue<Tensor>]
AsyncValue* AddWrapper(const std::vector<AsyncValue*>& args, HostContext* host) {
  // 1. Unwrap Inputs (Blocking for simplicity in this wrapper, 
  //    but practically they are already "Available" effectively if dispatched)
  //    In a pure async world, we might register a callback. 
  //    But ExecutionEngine usually ensures inputs are ready before dispatching this closure.
  
  // Cast raw void* back to Tensor* (Assuming we stored Tensor* in the AsyncValue)
  // Phase 2 ResolveArg stored `void*` -> we cast to Tensor*
  
  if (args.size() != 2) return host->MakeErrorAsyncValue("Add requires 2 arguments");
  
  // NOTE: This assumes the AsyncValues contain `Tensor*` or `Tensor`.
  // Our ExecutionEngine stores `void*`. 
  // Let's assume the void* points to a Tensor object managed elsewhere or passed in.
  
  auto* lhs_av = dynamic_cast<ConcreteAsyncValue<void*>*>(args[0]);
  auto* rhs_av = dynamic_cast<ConcreteAsyncValue<void*>*>(args[1]);
  
  if (!lhs_av || !rhs_av) return host->MakeErrorAsyncValue("Invalid arguments");
  
  Tensor* lhs = static_cast<Tensor*>(lhs_av->get());
  Tensor* rhs = static_cast<Tensor*>(rhs_av->get());
  
  // 2. Call Native Kernel
  // Using OwnTensor operator overload or function
  // Tensor result = *lhs + *rhs; 
  // For safety without full headers, let's just print or mock if headers missing
  
  // Silence unused variable warnings for scaffold
  (void)lhs; 
  (void)rhs;

#if defined(HAS_TENSORLIB)
  // Actual call
   Tensor* result_tensor = new Tensor(*lhs + *rhs);
#else
  // Scaffold call
  // std::cout << "Executing Host Add (Mock)\n";
  Tensor* result_tensor = new Tensor(); 
#endif

  // 3. Wrap Result
  return host->MakeAvailableAsyncValue<void*>(result_tensor);
}

// --- Binary Operations ---

AsyncValue* SubWrapper(const std::vector<AsyncValue*>& args, HostContext* host) {
  if (args.size() != 2) return host->MakeErrorAsyncValue("Sub requires 2 arguments");
  
  auto* lhs_av = dynamic_cast<ConcreteAsyncValue<void*>*>(args[0]);
  auto* rhs_av = dynamic_cast<ConcreteAsyncValue<void*>*>(args[1]);
  
  if (!lhs_av || !rhs_av) return host->MakeErrorAsyncValue("Invalid arguments");
  
  Tensor* lhs = static_cast<Tensor*>(lhs_av->get());
  Tensor* rhs = static_cast<Tensor*>(rhs_av->get());
  
#if defined(HAS_TENSORLIB)
  Tensor* result_tensor = new Tensor(*lhs - *rhs);
#else
  (void)lhs; (void)rhs;
  Tensor* result_tensor = new Tensor();
#endif
  
  return host->MakeAvailableAsyncValue<void*>(result_tensor);
}

AsyncValue* MulWrapper(const std::vector<AsyncValue*>& args, HostContext* host) {
  if (args.size() != 2) return host->MakeErrorAsyncValue("Mul requires 2 arguments");
  
  auto* lhs_av = dynamic_cast<ConcreteAsyncValue<void*>*>(args[0]);
  auto* rhs_av = dynamic_cast<ConcreteAsyncValue<void*>*>(args[1]);
  
  if (!lhs_av || !rhs_av) return host->MakeErrorAsyncValue("Invalid arguments");
  
  Tensor* lhs = static_cast<Tensor*>(lhs_av->get());
  Tensor* rhs = static_cast<Tensor*>(rhs_av->get());
  
#if defined(HAS_TENSORLIB)
  Tensor* result_tensor = new Tensor(*lhs * *rhs);
#else
  (void)lhs; (void)rhs;
  Tensor* result_tensor = new Tensor();
#endif
  
  return host->MakeAvailableAsyncValue<void*>(result_tensor);
}

AsyncValue* MatMulWrapper(const std::vector<AsyncValue*>& args, HostContext* host) {
  if (args.size() != 2) return host->MakeErrorAsyncValue("MatMul requires 2 arguments");
  
  auto* lhs_av = dynamic_cast<ConcreteAsyncValue<void*>*>(args[0]);
  auto* rhs_av = dynamic_cast<ConcreteAsyncValue<void*>*>(args[1]);
  
  if (!lhs_av || !rhs_av) return host->MakeErrorAsyncValue("Invalid arguments");
  
  Tensor* lhs = static_cast<Tensor*>(lhs_av->get());
  Tensor* rhs = static_cast<Tensor*>(rhs_av->get());
  
#if defined(HAS_TENSORLIB)
  Tensor* result_tensor = new Tensor(OwnTensor::matmul(*lhs, *rhs));
#else
  (void)lhs; (void)rhs;
  Tensor* result_tensor = new Tensor();
#endif
  
  return host->MakeAvailableAsyncValue<void*>(result_tensor);
}

// --- Unary Operations ---

AsyncValue* ExpWrapper(const std::vector<AsyncValue*>& args, HostContext* host) {
  if (args.size() != 1) return host->MakeErrorAsyncValue("Exp requires 1 argument");
  
  auto* input_av = dynamic_cast<ConcreteAsyncValue<void*>*>(args[0]);
  if (!input_av) return host->MakeErrorAsyncValue("Invalid argument");
  
  Tensor* input = static_cast<Tensor*>(input_av->get());
  
#if defined(HAS_TENSORLIB)
  Tensor* result_tensor = new Tensor(OwnTensor::exp(*input));
#else
  (void)input;
  Tensor* result_tensor = new Tensor();
#endif
  
  return host->MakeAvailableAsyncValue<void*>(result_tensor);
}

AsyncValue* LogWrapper(const std::vector<AsyncValue*>& args, HostContext* host) {
  if (args.size() != 1) return host->MakeErrorAsyncValue("Log requires 1 argument");
  
  auto* input_av = dynamic_cast<ConcreteAsyncValue<void*>*>(args[0]);
  if (!input_av) return host->MakeErrorAsyncValue("Invalid argument");
  
  Tensor* input = static_cast<Tensor*>(input_av->get());
  
#if defined(HAS_TENSORLIB)
  Tensor* result_tensor = new Tensor(OwnTensor::log(*input));
#else
  (void)input;
  Tensor* result_tensor = new Tensor();
#endif
  
  return host->MakeAvailableAsyncValue<void*>(result_tensor);
}

AsyncValue* TanhWrapper(const std::vector<AsyncValue*>& args, HostContext* host) {
  if (args.size() != 1) return host->MakeErrorAsyncValue("Tanh requires 1 argument");
  
  auto* input_av = dynamic_cast<ConcreteAsyncValue<void*>*>(args[0]);
  if (!input_av) return host->MakeErrorAsyncValue("Invalid argument");
  
  Tensor* input = static_cast<Tensor*>(input_av->get());
  
#if defined(HAS_TENSORLIB)
  Tensor* result_tensor = new Tensor(OwnTensor::tanh(*input));
#else
  (void)input;
  Tensor* result_tensor = new Tensor();
#endif
  
  return host->MakeAvailableAsyncValue<void*>(result_tensor);
}

AsyncValue* ReluWrapper(const std::vector<AsyncValue*>& args, HostContext* host) {
  if (args.size() != 1) return host->MakeErrorAsyncValue("Relu requires 1 argument");
  
  auto* input_av = dynamic_cast<ConcreteAsyncValue<void*>*>(args[0]);
  if (!input_av) return host->MakeErrorAsyncValue("Invalid argument");
  
  Tensor* input = static_cast<Tensor*>(input_av->get());
  
#if defined(HAS_TENSORLIB)
  // ReLU: max(0, x) or using the formula from jit_compiler: (x + abs(x)) * 0.5
  Tensor* result_tensor = new Tensor((*input + OwnTensor::abs(*input)) * 0.5f);
#else
  (void)input;
  Tensor* result_tensor = new Tensor();
#endif
  
  return host->MakeAvailableAsyncValue<void*>(result_tensor);
}

// --- Reduction Operations ---

AsyncValue* SumWrapper(const std::vector<AsyncValue*>& args, HostContext* host) {
  if (args.size() != 1) return host->MakeErrorAsyncValue("Sum requires 1 argument");
  
  auto* input_av = dynamic_cast<ConcreteAsyncValue<void*>*>(args[0]);
  if (!input_av) return host->MakeErrorAsyncValue("Invalid argument");
  
  Tensor* input = static_cast<Tensor*>(input_av->get());
  
#if defined(HAS_TENSORLIB)
  Tensor* result_tensor = new Tensor(OwnTensor::reduce_sum(*input));
#else
  (void)input;
  Tensor* result_tensor = new Tensor();
#endif
  
  return host->MakeAvailableAsyncValue<void*>(result_tensor);
}

AsyncValue* MeanWrapper(const std::vector<AsyncValue*>& args, HostContext* host) {
  if (args.size() != 1) return host->MakeErrorAsyncValue("Mean requires 1 argument");
  
  auto* input_av = dynamic_cast<ConcreteAsyncValue<void*>*>(args[0]);
  if (!input_av) return host->MakeErrorAsyncValue("Invalid argument");
  
  Tensor* input = static_cast<Tensor*>(input_av->get());
  
#if defined(HAS_TENSORLIB)
  Tensor* result_tensor = new Tensor(OwnTensor::reduce_mean(*input));
#else
  (void)input;
  Tensor* result_tensor = new Tensor();
#endif
  
  return host->MakeAvailableAsyncValue<void*>(result_tensor);
}

AsyncValue* MaxWrapper(const std::vector<AsyncValue*>& args, HostContext* host) {
  if (args.size() != 1) return host->MakeErrorAsyncValue("Max requires 1 argument");
  
  auto* input_av = dynamic_cast<ConcreteAsyncValue<void*>*>(args[0]);
  if (!input_av) return host->MakeErrorAsyncValue("Invalid argument");
  
  Tensor* input = static_cast<Tensor*>(input_av->get());
  
#if defined(HAS_TENSORLIB)
  Tensor* result_tensor = new Tensor(OwnTensor::reduce_max(*input));
#else
  (void)input;
  Tensor* result_tensor = new Tensor();
#endif
  
  return host->MakeAvailableAsyncValue<void*>(result_tensor);
}

// --- Registration ---

void RegisterHostKernels(KernelRegistry& registry) {
  // Binary operations
  registry.RegisterKernel("nova.add", Device::CPU, AddWrapper);
  registry.RegisterKernel("nova.sub", Device::CPU, SubWrapper);
  registry.RegisterKernel("nova.mul", Device::CPU, MulWrapper);
  registry.RegisterKernel("nova.matmul", Device::CPU, MatMulWrapper);
  
  // Unary operations
  registry.RegisterKernel("nova.exp", Device::CPU, ExpWrapper);
  registry.RegisterKernel("nova.log", Device::CPU, LogWrapper);
  registry.RegisterKernel("nova.tanh", Device::CPU, TanhWrapper);
  registry.RegisterKernel("nova.relu", Device::CPU, ReluWrapper);
  
  // Reduction operations
  registry.RegisterKernel("nova.sum", Device::CPU, SumWrapper);
  registry.RegisterKernel("nova.mean", Device::CPU, MeanWrapper);
  registry.RegisterKernel("nova.max", Device::CPU, MaxWrapper);
}

} // namespace runtime
} // namespace nova
