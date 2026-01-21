//===- AsyncValue.h - Core Primitive for Async Execution --------*- C++ -*-===//
//
// Nova Compiler Runtime
//
//===----------------------------------------------------------------------===//

#ifndef NOVA_RUNTIME_CORE_ASYNCVALUE_H
#define NOVA_RUNTIME_CORE_ASYNCVALUE_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <variant>
#include <vector>

namespace nova {
namespace runtime {

class HostContext;

// Base class for AsyncValue to allow type-erased handling
class AsyncValue {
public:
  enum class State { Unconstructed, Concrete, Error };

  virtual ~AsyncValue() = default;

  State GetState() const { return state_.load(std::memory_order_acquire); }
  bool IsAvailable() const { return GetState() == State::Concrete; }
  bool IsError() const { return GetState() == State::Error; }
  bool IsUnconstructed() const { return GetState() == State::Unconstructed; }

  // Block until the value is available or error.
  void Await();

  // If available, executes callback immediately.
  // Otherwise, adds to waiter list.
  void AndThen(std::function<void(AsyncValue *)> callback);

  std::string GetError() const;
  void SetError(std::string error_msg);

protected:
  AsyncValue(State state = State::Unconstructed) : state_(state) {}

  void NotifyAvailable(State new_state);

  std::atomic<State> state_;
  std::string error_message_; // Only valid if state_ == Error
  mutable std::mutex mutex_;
  mutable std::condition_variable cv_;
  std::vector<std::function<void(AsyncValue *)>> waiters_;

friend class HostContext; 
};

template <typename T> class ConcreteAsyncValue : public AsyncValue {
public:
  ConcreteAsyncValue() : AsyncValue(State::Unconstructed) {}

  // Emplace the value, making it available.
  template <typename... Args> void emplace(Args &&...args) {
    new (&storage_) T(std::forward<Args>(args)...);
    NotifyAvailable(State::Concrete);
  }

  T &get() { return *reinterpret_cast<T *>(&storage_); }
  const T &get() const { return *reinterpret_cast<const T *>(&storage_); }

private:
  alignas(T) std::byte storage_[sizeof(T)];
};

// Specialization for void (no storage needed)
template <> class ConcreteAsyncValue<void> : public AsyncValue {
public:
  ConcreteAsyncValue() : AsyncValue(State::Unconstructed) {}

  // For void, emplace just marks as available
  void emplace() {
    NotifyAvailable(State::Concrete);
  }
};

} // namespace runtime
} // namespace nova

#endif // NOVA_RUNTIME_CORE_ASYNCVALUE_H
