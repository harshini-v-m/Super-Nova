//===- KernelRegistry.h - Kernel Registration and Dispatch ---------*- C++ -*-===//
//
// Nova Compiler Runtime
//
//===----------------------------------------------------------------------===//

#ifndef NOVA_RUNTIME_EXECUTOR_KERNELREGISTRY_H
#define NOVA_RUNTIME_EXECUTOR_KERNELREGISTRY_H

#include "Runtime/Core/AsyncValue.h"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace nova {
namespace runtime {

class HostContext;

// Forward declaration for Tensor (will come from OwnTensor)
// For now, we use a placeholder or assume it will be available
namespace detail {
  class TensorPlaceholder; // Replace with actual Tensor type when linking
}

// Kernel function signature:
// Takes a vector of AsyncValue<Tensor>* inputs and returns AsyncValue<Tensor>*
using KernelFunction = std::function<AsyncValue*(
    const std::vector<AsyncValue*>& inputs,
    HostContext* host)>;

// Device type enum (matches OwnTensor::Device)
enum class Device {
  CPU = 0,
  CUDA = 1
};

class KernelRegistry {
public:
  // Get the singleton instance
  static KernelRegistry& Instance();

  // Register a kernel for a specific operation and device
  void RegisterKernel(const std::string& op_name,
                      Device device,
                      KernelFunction kernel);

  // Lookup a kernel by operation name and device
  KernelFunction GetKernel(const std::string& op_name,
                           Device device) const;

  // Check if a kernel exists
  bool HasKernel(const std::string& op_name, Device device) const;

private:
  KernelRegistry() = default;
  ~KernelRegistry() = default;

  // Prevent copying
  KernelRegistry(const KernelRegistry&) = delete;
  KernelRegistry& operator=(const KernelRegistry&) = delete;

  struct KernelKey {
    std::string op_name;
    Device device;

    bool operator==(const KernelKey& other) const {
      return op_name == other.op_name && device == other.device;
    }
  };

  struct KernelKeyHash {
    std::size_t operator()(const KernelKey& key) const {
      return std::hash<std::string>()(key.op_name) ^
             (std::hash<int>()(static_cast<int>(key.device)) << 1);
    }
  };

  std::unordered_map<KernelKey, KernelFunction, KernelKeyHash> kernels_;
};

// Helper macro for kernel registration
#define REGISTER_KERNEL(OpName, DeviceType, KernelFunc) \
  namespace { \
    struct KernelRegistration_##OpName##_##DeviceType { \
      KernelRegistration_##OpName##_##DeviceType() { \
        ::nova::runtime::KernelRegistry::Instance().RegisterKernel( \
            #OpName, ::nova::runtime::Device::DeviceType, KernelFunc); \
      } \
    }; \
    static KernelRegistration_##OpName##_##DeviceType \
        kernel_registration_##OpName##_##DeviceType; \
  }

} // namespace runtime
} // namespace nova

#endif // NOVA_RUNTIME_EXECUTOR_KERNELREGISTRY_H
