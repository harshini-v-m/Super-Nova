//===- MemoryAllocator.h - Memory Management -----------------------*- C++ -*-===//
//
// Nova Compiler Runtime
//
//===----------------------------------------------------------------------===//

#ifndef NOVA_RUNTIME_CORE_MEMORYALLOCATOR_H
#define NOVA_RUNTIME_CORE_MEMORYALLOCATOR_H

#include <cstddef>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <memory>

// Forward declarations from OwnTensor (assuming we can link to it)
// We use void* or int for device index to avoid strict dependency in header if possible,
// but for now let's assume we can include the types or forward declare.
// Since existing code uses DeviceIndex, let's look for it.
// It seems DeviceIndex is a struct in TensorLib.h.
// For the runtime, we might want to abstract it or just duplicate the simple struct if separate.
// But to wrap it, we should probably include it.
// However, currently we are in Nova-Compiler/include. It might not know about TensorLib yet.
// I'll use int for DeviceIndex in the interface for now to keep it decoupled at header level, or use a matching type.
// User prompt used DeviceIndex. I'll define a compatible alias or use int.

namespace nova {
namespace runtime {

using DeviceIndex = int; // Placeholder, maps to OwnTensor::DeviceIndex::index

class AsyncValueAllocator {
public:
  void *Allocate(size_t size, size_t alignment);
  void Deallocate(void *ptr);

private:
  // Simple free list or slab could go here.
  // For Phase 1, we might just use malloc/free for simplicity.
  std::mutex mutex_;
};

class DeviceMemoryManager {
public:
  DeviceMemoryManager();
  ~DeviceMemoryManager();

  void *AllocateDevice(size_t bytes, DeviceIndex device);
  void DeallocateDevice(void *ptr, DeviceIndex device);

  // Buffer recycling
  void *GetOrAllocate(size_t bytes, DeviceIndex device);
  void Recycle(void *ptr, size_t bytes, DeviceIndex device);

private:
  struct BufferPool;
  std::unordered_map<DeviceIndex, std::unique_ptr<BufferPool>> pools_;
  std::mutex mutex_;
};

} // namespace runtime
} // namespace nova

#endif // NOVA_RUNTIME_CORE_MEMORYALLOCATOR_H
