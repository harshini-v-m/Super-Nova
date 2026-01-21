//===- KernelRegistration.h - Kernel Registration Logic -------------*- C++ -*-===//
//
// Nova Compiler Runtime
//
//===----------------------------------------------------------------------===//

#ifndef NOVA_RUNTIME_KERNELS_KERNELREGISTRATION_H
#define NOVA_RUNTIME_KERNELS_KERNELREGISTRATION_H

#include "Runtime/Executor/KernelRegistry.h"

namespace nova {
namespace runtime {

// Register all available kernels (Host and CUDA) with the registry.
void RegisterAllKernels(KernelRegistry& registry);

// Register only Host kernels
void RegisterHostKernels(KernelRegistry& registry);

// Register only CUDA kernels (if available)
void RegisterCudaKernels(KernelRegistry& registry);

} // namespace runtime
} // namespace nova

#endif // NOVA_RUNTIME_KERNELS_KERNELREGISTRATION_H
