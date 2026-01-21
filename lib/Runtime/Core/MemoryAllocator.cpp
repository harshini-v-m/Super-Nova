//===- MemoryAllocator.cpp - Memory Management ----------------------------===//
//
// Nova Compiler Runtime
//
//===----------------------------------------------------------------------===//

#include "Runtime/Core/MemoryAllocator.h"
#include <cstdlib>
#include <iostream>

// We need to include OwnTensor headers to actually allocate device memory.
// Assuming the build system will make them available.
// If not, this file will fail to compile later, but we are writing the "Scaffolding".
// #include "device/AllocatorRegistry.h" 
// #include "device/Allocator.h"
// #include "device/Device.h"

namespace nova {
namespace runtime {

void *AsyncValueAllocator::Allocate(size_t size, size_t alignment) {
  // Simple aligned alloc
  return std::aligned_alloc(alignment, size);
}

void AsyncValueAllocator::Deallocate(void *ptr) {
  std::free(ptr);
}

// --- DeviceMemoryManager ---

struct DeviceMemoryManager::BufferPool {
    // Simple bucketed pool could go here
    std::vector<std::pair<void*, size_t>> free_buffers;
};

DeviceMemoryManager::DeviceMemoryManager() = default;
DeviceMemoryManager::~DeviceMemoryManager() = default;

void *DeviceMemoryManager::AllocateDevice(size_t bytes, DeviceIndex device) {
  // Implementation will delegate to OwnTensor::AllocatorRegistry
  // For scaffolding, we'll just log or return nullptr if not linked.
  // TODO: Link with OwnTensor
  // OwnTensor::Allocator* allocator = OwnTensor::AllocatorRegistry::get_allocator(OwnTensor::Device(OwnTensor::Device::Type(device)));
  // return allocator->allocate(bytes);
  return nullptr; 
}

void DeviceMemoryManager::DeallocateDevice(void *ptr, DeviceIndex device) {
  // TODO: Link with OwnTensor
  // OwnTensor::Allocator* allocator = ...
  // allocator->deallocate(ptr);
}

void *DeviceMemoryManager::GetOrAllocate(size_t bytes, DeviceIndex device) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pools_.find(device) == pools_.end()) {
        pools_[device] = std::make_unique<BufferPool>();
    }
    
    // Check pool (ommitted for brevity/scaffolding)
    
    return AllocateDevice(bytes, device);
}

void DeviceMemoryManager::Recycle(void *ptr, size_t bytes, DeviceIndex device) {
     std::lock_guard<std::mutex> lock(mutex_);
     if (pools_.find(device) == pools_.end()) {
        pools_[device] = std::make_unique<BufferPool>();
    }
    pools_[device]->free_buffers.push_back({ptr, bytes});
}

} // namespace runtime
} // namespace nova
