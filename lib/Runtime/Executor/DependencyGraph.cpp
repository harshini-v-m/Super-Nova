//===- DependencyGraph.cpp - Task Dependency Tracking ---------------------===//
//
// Nova Compiler Runtime
//
//===----------------------------------------------------------------------===//

#include "Runtime/Executor/DependencyGraph.h"
#include <algorithm>
#include <stdexcept>

namespace nova {
namespace runtime {

void DependencyGraph::AddTask(int task_id, const std::vector<int>& dependencies) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  // Ensure we have enough space
  if (task_id >= static_cast<int>(nodes_.size())) {
    nodes_.resize(task_id + 1);
  }
  
  auto& node = nodes_[task_id];
  node.task_id = task_id;
  node.dependencies = dependencies;
  node.pending_deps = dependencies.size();
  node.completed = false;
  
  // Update dependents for each dependency
  for (int dep_id : dependencies) {
    if (dep_id >= static_cast<int>(nodes_.size())) {
      nodes_.resize(dep_id + 1);
    }
    nodes_[dep_id].dependents.push_back(task_id);
  }
}

std::vector<int> DependencyGraph::GetReadyTasks() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<int> ready;
  
  for (const auto& node : nodes_) {
    if (!node.completed && node.pending_deps == 0 && node.task_id >= 0) {
      ready.push_back(node.task_id);
    }
  }
  
  return ready;
}

void DependencyGraph::MarkComplete(int task_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  if (task_id >= static_cast<int>(nodes_.size())) {
    throw std::runtime_error("Invalid task_id in MarkComplete");
  }
  
  auto& node = nodes_[task_id];
  
  // Mark as completed
  if (node.completed) {
    return; // Already completed
  }
  
  node.completed = true;
  completed_count_++;
  
  // Decrement pending_deps for all dependents
  for (int dependent_id : node.dependents) {
    nodes_[dependent_id].pending_deps--;
  }
}

bool DependencyGraph::IsComplete() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return completed_count_ == static_cast<int>(nodes_.size());
}

void DependencyGraph::Reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  
  for (auto& node : nodes_) {
    node.pending_deps = node.dependencies.size();
    node.completed = false;
  }
  
  completed_count_ = 0;
}

} // namespace runtime
} // namespace nova
