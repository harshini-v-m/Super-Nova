//===----------------------------------------------------------------------===//
// 
// Nova Compiler Runtime
// 
//===----------------------------------------------------------------------===//

#include "Runtime/Core/AsyncValue.h"
#include <cassert>

namespace nova {
namespace runtime {

void AsyncValue::Await() {
  if (IsAvailable() || IsError())
    return;

  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return IsAvailable() || IsError(); });
}

void AsyncValue::AndThen(std::function<void(AsyncValue *)> callback) {
  // Optimization: type check state without lock first
  if (IsAvailable() || IsError()) {
    callback(this);
    return;
  }

  std::unique_lock<std::mutex> lock(mutex_);
  if (IsAvailable() || IsError()) {
    lock.unlock();
    callback(this);
    return;
  }
  waiters_.push_back(std::move(callback));
}

std::string AsyncValue::GetError() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return error_message_;
}

void AsyncValue::SetError(std::string error_msg) {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (state_ != State::Unconstructed) {
      assert(false && "Cannot set error on a non-empty AsyncValue");
      return;
    }
    error_message_ = std::move(error_msg);
  }
  NotifyAvailable(State::Error);
}

void AsyncValue::NotifyAvailable(State new_state) {
  // Capture waiters to run outside the lock
  std::vector<std::function<void(AsyncValue *)>> local_waiters;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    state_.store(new_state, std::memory_order_release);
    local_waiters.swap(waiters_);
  }
  cv_.notify_all();

  for (auto &waiter : local_waiters) {
    waiter(this);
  }
}

} // namespace runtime
} // namespace nova
