//===- HostContext.h - Runtime Environment Manager -----------------*- C++ -*-===//
//
// Nova Compiler Runtime
//
//===----------------------------------------------------------------------===//

#ifndef NOVA_RUNTIME_CORE_HOSTCONTEXT_H
#define NOVA_RUNTIME_CORE_HOSTCONTEXT_H

#include "Runtime/Core/AsyncValue.h"
#include <atomic>
#include <functional>
#include <memory>
#include <vector>

namespace nova {
namespace runtime {

class ThreadPool;

class HostContext {
public:
  // num_threads: -1 means use hardware concurrency.
  explicit HostContext(int num_threads = -1);
  ~HostContext();

  // Enqueue a task to run asynchronously.
  void EnqueueWork(std::function<void()> work);

  // Enqueue a task that might block behavior (separate handling if needed).
  void EnqueueBlockingWork(std::function<void()> work);

  // --- AsyncValue Allocation ---

  // Create an available AsyncValue<T>
  template <typename T, typename... Args>
  ConcreteAsyncValue<T> *MakeAvailableAsyncValue(Args &&...args) {
    auto *av = new ConcreteAsyncValue<T>();
    av->emplace(std::forward<Args>(args)...);
    return av;
  }

  // Create an unconstructed AsyncValue<T>
  template <typename T> ConcreteAsyncValue<T> *MakeUnconstructedAsyncValue() {
    return new ConcreteAsyncValue<T>();
  }

  // Create an Error AsyncValue
  AsyncValue *MakeErrorAsyncValue(std::string message);

  int GetNumThreads() const;

private:
  std::unique_ptr<ThreadPool> thread_pool_;
};

} // namespace runtime
} // namespace nova

#endif // NOVA_RUNTIME_CORE_HOSTCONTEXT_H
