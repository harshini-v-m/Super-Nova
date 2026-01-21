//===----------------------------------------------------------------------===//
//
// Nova Compiler Runtime
//
//===----------------------------------------------------------------------===//

#include "Runtime/Core/HostContext.h"
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

namespace nova {
namespace runtime {

// Simple ThreadPool implementation
class ThreadPool {
public:
  ThreadPool(int num_threads) : stop_(false) {
    if (num_threads < 1) {
      num_threads = std::thread::hardware_concurrency();
      if (num_threads < 1)
        num_threads = 1;
    }

    for (int i = 0; i < num_threads; ++i) {
      workers_.emplace_back([this] {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });

            if (stop_ && tasks_.empty())
              return;

            task = std::move(tasks_.front());
            tasks_.pop();
          }
          task();
        }
      });
    }
  }

  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      stop_ = true;
    }
    condition_.notify_all();
    for (std::thread &worker : workers_) {
      if (worker.joinable())
        worker.join();
    }
  }

  void Enqueue(std::function<void()> task) {
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      tasks_.push(std::move(task));
    }
    condition_.notify_one();
  }

  int size() const { return workers_.size(); }

private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex queue_mutex_;
  std::condition_variable condition_;
  bool stop_;
};

// --- HostContext Implementation ---

HostContext::HostContext(int num_threads) {
  thread_pool_ = std::make_unique<ThreadPool>(num_threads);
}

HostContext::~HostContext() = default;

void HostContext::EnqueueWork(std::function<void()> work) {
  thread_pool_->Enqueue(std::move(work));
}

void HostContext::EnqueueBlockingWork(std::function<void()> work) {
  // For now, treat blocking work same as regular work.
  // In future, this could use a separate thread pool to avoid starvation.
  thread_pool_->Enqueue(std::move(work));
}

AsyncValue *HostContext::MakeErrorAsyncValue(std::string message) {
  auto *av = new AsyncValue(AsyncValue::State::Unconstructed);
  av->SetError(std::move(message));
  return av;
}

int HostContext::GetNumThreads() const { return thread_pool_->size(); }

} // namespace runtime
} // namespace nova
