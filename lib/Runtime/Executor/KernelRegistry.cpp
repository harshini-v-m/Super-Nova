//===- KernelRegistry.cpp - Kernel Registration and Dispatch --------------===//
//
// Nova Compiler Runtime
//
//===----------------------------------------------------------------------===//

#include "Runtime/Executor/KernelRegistry.h"
#include <iostream>
#include <stdexcept>

namespace nova {
namespace runtime {

KernelRegistry& KernelRegistry::Instance() {
  static KernelRegistry instance;
  return instance;
}

void KernelRegistry::RegisterKernel(const std::string& op_name,
                                    Device device,
                                    KernelFunction kernel) {
  KernelKey key{op_name, device};
  
  if (kernels_.find(key) != kernels_.end()) {
    std::cerr << "Warning: Overwriting kernel for op '" << op_name 
              << "' on device " << static_cast<int>(device) << "\n";
  }
  
  kernels_[key] = std::move(kernel);
}

KernelFunction KernelRegistry::GetKernel(const std::string& op_name,
                                         Device device) const {
  KernelKey key{op_name, device};
  auto it = kernels_.find(key);
  
  if (it == kernels_.end()) {
    throw std::runtime_error("Kernel not found for op '" + op_name + 
                             "' on device " + std::to_string(static_cast<int>(device)));
  }
  
  return it->second;
}

bool KernelRegistry::HasKernel(const std::string& op_name, Device device) const {
  KernelKey key{op_name, device};
  return kernels_.find(key) != kernels_.end();
}

} // namespace runtime
} // namespace nova
