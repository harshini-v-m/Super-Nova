//===- CudaKernels.cpp - GPU Kernel Wrappers ------------------------------===//
//
// Nova Compiler Runtime
//
//===----------------------------------------------------------------------===//

#include "Runtime/Kernels/KernelRegistration.h"
#include "Runtime/Core/AsyncValue.h"
#include "Runtime/Core/HostContext.h"
#include <iostream>

// --- TensorLib Integration ---
#if __has_include("core/Tensor.h") || defined(HAS_TENSORLIB)
#include "core/Tensor.h"
#include "ops/TensorOps.h"
#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif
using Tensor = OwnTensor::Tensor;
#else
// Mock Tensor for standalone compilation
namespace OwnTensor {
  struct Tensor { 
    static Tensor zeros(std::vector<long>, bool) { return {}; }
  }; 
}
using Tensor = OwnTensor::Tensor;
#endif

namespace nova {
namespace runtime {

// Helper to get current CUDA stream (from cgadimpl runtime)
// For now, use default stream (nullptr)
#ifdef WITH_CUDA
inline cudaStream_t GetCurrentStream() {
  // In real integration, this would call ag::current_stream()
  return 0; // Default stream
}
#else
inline void* GetCurrentStream() {
  return nullptr;
}
#endif

// --- Binary Operations ---

AsyncValue* CudaAddWrapper(const std::vector<AsyncValue*>& args, HostContext* host) {
  if (args.size() != 2) return host->MakeErrorAsyncValue("CUDA Add requires 2 arguments");
  
  auto* lhs_av = dynamic_cast<ConcreteAsyncValue<void*>*>(args[0]);
  auto* rhs_av = dynamic_cast<ConcreteAsyncValue<void*>*>(args[1]);
  
  if (!lhs_av || !rhs_av) return host->MakeErrorAsyncValue("Invalid arguments");
  
  Tensor* lhs = static_cast<Tensor*>(lhs_av->get());
  Tensor* rhs = static_cast<Tensor*>(rhs_av->get());
  
#if defined(HAS_TENSORLIB) && defined(WITH_CUDA)
  cudaStream_t stream = GetCurrentStream();
  Tensor* result_tensor = new Tensor(*lhs + *rhs); // Operator uses CUDA if tensors are on GPU
#else
  (void)lhs; (void)rhs;
  Tensor* result_tensor = new Tensor();
#endif
  
  return host->MakeAvailableAsyncValue<void*>(result_tensor);
}

AsyncValue* CudaSubWrapper(const std::vector<AsyncValue*>& args, HostContext* host) {
  if (args.size() != 2) return host->MakeErrorAsyncValue("CUDA Sub requires 2 arguments");
  
  auto* lhs_av = dynamic_cast<ConcreteAsyncValue<void*>*>(args[0]);
  auto* rhs_av = dynamic_cast<ConcreteAsyncValue<void*>*>(args[1]);
  
  if (!lhs_av || !rhs_av) return host->MakeErrorAsyncValue("Invalid arguments");
  
  Tensor* lhs = static_cast<Tensor*>(lhs_av->get());
  Tensor* rhs = static_cast<Tensor*>(rhs_av->get());
  
#if defined(HAS_TENSORLIB) && defined(WITH_CUDA)
  Tensor* result_tensor = new Tensor(*lhs - *rhs);
#else
  (void)lhs; (void)rhs;
  Tensor* result_tensor = new Tensor();
#endif
  
  return host->MakeAvailableAsyncValue<void*>(result_tensor);
}

AsyncValue* CudaMulWrapper(const std::vector<AsyncValue*>& args, HostContext* host) {
  if (args.size() != 2) return host->MakeErrorAsyncValue("CUDA Mul requires 2 arguments");
  
  auto* lhs_av = dynamic_cast<ConcreteAsyncValue<void*>*>(args[0]);
  auto* rhs_av = dynamic_cast<ConcreteAsyncValue<void*>*>(args[1]);
  
  if (!lhs_av || !rhs_av) return host->MakeErrorAsyncValue("Invalid arguments");
  
  Tensor* lhs = static_cast<Tensor*>(lhs_av->get());
  Tensor* rhs = static_cast<Tensor*>(rhs_av->get());
  
#if defined(HAS_TENSORLIB) && defined(WITH_CUDA)
  Tensor* result_tensor = new Tensor(*lhs * *rhs);
#else
  (void)lhs; (void)rhs;
  Tensor* result_tensor = new Tensor();
#endif
  
  return host->MakeAvailableAsyncValue<void*>(result_tensor);
}

AsyncValue* CudaMatMulWrapper(const std::vector<AsyncValue*>& args, HostContext* host) {
  if (args.size() != 2) return host->MakeErrorAsyncValue("CUDA MatMul requires 2 arguments");
  
  auto* lhs_av = dynamic_cast<ConcreteAsyncValue<void*>*>(args[0]);
  auto* rhs_av = dynamic_cast<ConcreteAsyncValue<void*>*>(args[1]);
  
  if (!lhs_av || !rhs_av) return host->MakeErrorAsyncValue("Invalid arguments");
  
  Tensor* lhs = static_cast<Tensor*>(lhs_av->get());
  Tensor* rhs = static_cast<Tensor*>(rhs_av->get());
  
#if defined(HAS_TENSORLIB) && defined(WITH_CUDA)
  cudaStream_t stream = GetCurrentStream();
  Tensor* result_tensor = new Tensor(OwnTensor::matmul(*lhs, *rhs, stream));
#else
  (void)lhs; (void)rhs;
  Tensor* result_tensor = new Tensor();
#endif
  
  return host->MakeAvailableAsyncValue<void*>(result_tensor);
}

// --- Unary Operations ---

AsyncValue* CudaExpWrapper(const std::vector<AsyncValue*>& args, HostContext* host) {
  if (args.size() != 1) return host->MakeErrorAsyncValue("CUDA Exp requires 1 argument");
  
  auto* input_av = dynamic_cast<ConcreteAsyncValue<void*>*>(args[0]);
  if (!input_av) return host->MakeErrorAsyncValue("Invalid argument");
  
  Tensor* input = static_cast<Tensor*>(input_av->get());
  
#if defined(HAS_TENSORLIB) && defined(WITH_CUDA)
  cudaStream_t stream = GetCurrentStream();
  Tensor* result_tensor = new Tensor(OwnTensor::exp(*input, stream));
#else
  (void)input;
  Tensor* result_tensor = new Tensor();
#endif
  
  return host->MakeAvailableAsyncValue<void*>(result_tensor);
}

AsyncValue* CudaLogWrapper(const std::vector<AsyncValue*>& args, HostContext* host) {
  if (args.size() != 1) return host->MakeErrorAsyncValue("CUDA Log requires 1 argument");
  
  auto* input_av = dynamic_cast<ConcreteAsyncValue<void*>*>(args[0]);
  if (!input_av) return host->MakeErrorAsyncValue("Invalid argument");
  
  Tensor* input = static_cast<Tensor*>(input_av->get());
  
#if defined(HAS_TENSORLIB) && defined(WITH_CUDA)
  cudaStream_t stream = GetCurrentStream();
  Tensor* result_tensor = new Tensor(OwnTensor::log(*input, stream));
#else
  (void)input;
  Tensor* result_tensor = new Tensor();
#endif
  
  return host->MakeAvailableAsyncValue<void*>(result_tensor);
}

AsyncValue* CudaTanhWrapper(const std::vector<AsyncValue*>& args, HostContext* host) {
  if (args.size() != 1) return host->MakeErrorAsyncValue("CUDA Tanh requires 1 argument");
  
  auto* input_av = dynamic_cast<ConcreteAsyncValue<void*>*>(args[0]);
  if (!input_av) return host->MakeErrorAsyncValue("Invalid argument");
  
  Tensor* input = static_cast<Tensor*>(input_av->get());
  
#if defined(HAS_TENSORLIB) && defined(WITH_CUDA)
  cudaStream_t stream = GetCurrentStream();
  Tensor* result_tensor = new Tensor(OwnTensor::tanh(*input, stream));
#else
  (void)input;
  Tensor* result_tensor = new Tensor();
#endif
  
  return host->MakeAvailableAsyncValue<void*>(result_tensor);
}

AsyncValue* CudaReluWrapper(const std::vector<AsyncValue*>& args, HostContext* host) {
  if (args.size() != 1) return host->MakeErrorAsyncValue("CUDA Relu requires 1 argument");
  
  auto* input_av = dynamic_cast<ConcreteAsyncValue<void*>*>(args[0]);
  if (!input_av) return host->MakeErrorAsyncValue("Invalid argument");
  
  Tensor* input = static_cast<Tensor*>(input_av->get());
  
#if defined(HAS_TENSORLIB) && defined(WITH_CUDA)
  cudaStream_t stream = GetCurrentStream();
  Tensor* result_tensor = new Tensor((*input + OwnTensor::abs(*input, stream)) * 0.5f);
#else
  (void)input;
  Tensor* result_tensor = new Tensor();
#endif
  
  return host->MakeAvailableAsyncValue<void*>(result_tensor);
}

// --- Reduction Operations ---

AsyncValue* CudaSumWrapper(const std::vector<AsyncValue*>& args, HostContext* host) {
  if (args.size() != 1) return host->MakeErrorAsyncValue("CUDA Sum requires 1 argument");
  
  auto* input_av = dynamic_cast<ConcreteAsyncValue<void*>*>(args[0]);
  if (!input_av) return host->MakeErrorAsyncValue("Invalid argument");
  
  Tensor* input = static_cast<Tensor*>(input_av->get());
  
#if defined(HAS_TENSORLIB) && defined(WITH_CUDA)
  Tensor* result_tensor = new Tensor(OwnTensor::reduce_sum(*input));
#else
  (void)input;
  Tensor* result_tensor = new Tensor();
#endif
  
  return host->MakeAvailableAsyncValue<void*>(result_tensor);
}

AsyncValue* CudaMeanWrapper(const std::vector<AsyncValue*>& args, HostContext* host) {
  if (args.size() != 1) return host->MakeErrorAsyncValue("CUDA Mean requires 1 argument");
  
  auto* input_av = dynamic_cast<ConcreteAsyncValue<void*>*>(args[0]);
  if (!input_av) return host->MakeErrorAsyncValue("Invalid argument");
  
  Tensor* input = static_cast<Tensor*>(input_av->get());
  
#if defined(HAS_TENSORLIB) && defined(WITH_CUDA)
  Tensor* result_tensor = new Tensor(OwnTensor::reduce_mean(*input));
#else
  (void)input;
  Tensor* result_tensor = new Tensor();
#endif
  
  return host->MakeAvailableAsyncValue<void*>(result_tensor);
}

AsyncValue* CudaMaxWrapper(const std::vector<AsyncValue*>& args, HostContext* host) {
  if (args.size() != 1) return host->MakeErrorAsyncValue("CUDA Max requires 1 argument");
  
  auto* input_av = dynamic_cast<ConcreteAsyncValue<void*>*>(args[0]);
  if (!input_av) return host->MakeErrorAsyncValue("Invalid argument");
  
  Tensor* input = static_cast<Tensor*>(input_av->get());
  
#if defined(HAS_TENSORLIB) && defined(WITH_CUDA)
  Tensor* result_tensor = new Tensor(OwnTensor::reduce_max(*input));
#else
  (void)input;
  Tensor* result_tensor = new Tensor();
#endif
  
  return host->MakeAvailableAsyncValue<void*>(result_tensor);
}

// --- Registration ---

void RegisterCudaKernels(KernelRegistry& registry) {
  // Binary operations
  registry.RegisterKernel("nova.add", Device::CUDA, CudaAddWrapper);
  registry.RegisterKernel("nova.sub", Device::CUDA, CudaSubWrapper);
  registry.RegisterKernel("nova.mul", Device::CUDA, CudaMulWrapper);
  registry.RegisterKernel("nova.matmul", Device::CUDA, CudaMatMulWrapper);
  
  // Unary operations
  registry.RegisterKernel("nova.exp", Device::CUDA, CudaExpWrapper);
  registry.RegisterKernel("nova.log", Device::CUDA, CudaLogWrapper);
  registry.RegisterKernel("nova.tanh", Device::CUDA, CudaTanhWrapper);
  registry.RegisterKernel("nova.relu", Device::CUDA, CudaReluWrapper);
  
  // Reduction operations
  registry.RegisterKernel("nova.sum", Device::CUDA, CudaSumWrapper);
  registry.RegisterKernel("nova.mean", Device::CUDA, CudaMeanWrapper);
  registry.RegisterKernel("nova.max", Device::CUDA, CudaMaxWrapper);
}

void RegisterAllKernels(KernelRegistry& registry) {
  RegisterHostKernels(registry);
  RegisterCudaKernels(registry);
}

} // namespace runtime
} // namespace nova

