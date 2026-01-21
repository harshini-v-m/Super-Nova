//===- DependencyGraph.h - Task Dependency Tracking ----------------*- C++ -*-===//
//
// Nova Compiler Runtime
//
//===----------------------------------------------------------------------===//

#ifndef NOVA_RUNTIME_EXECUTOR_DEPENDENCYGRAPH_H
#define NOVA_RUNTIME_EXECUTOR_DEPENDENCYGRAPH_H

#include <atomic>
#include <mutex>
#include <vector>

namespace nova {
namespace runtime {

class DependencyGraph {
public:
  DependencyGraph() = default;
  
  // Add a task with its dependencies
  void AddTask(int task_id, const std::vector<int>& dependencies);
  
  // Get tasks that are ready to execute (all dependencies satisfied)
  std::vector<int> GetReadyTasks() const;
  
  // Mark a task as complete and update dependent tasks
  void MarkComplete(int task_id);
  
  // Check if all tasks are complete
  bool IsComplete() const;
  
  // Reset the graph for re-execution
  void Reset();
  
  // Get number of tasks
  int GetNumTasks() const { return nodes_.size(); }

private:
  struct TaskNode {
    int task_id = -1;
    std::vector<int> dependencies;
    std::vector<int> dependents; // Tasks that depend on this one
    int pending_deps = 0;
    bool completed = false;
  };
  
  std::vector<TaskNode> nodes_;
  mutable std::mutex mutex_;
  int completed_count_ = 0;
};

} // namespace runtime
} // namespace nova

#endif // NOVA_RUNTIME_EXECUTOR_DEPENDENCYGRAPH_H
