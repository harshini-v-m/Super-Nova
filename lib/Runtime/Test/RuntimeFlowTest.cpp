//===- RuntimeFlowTest.cpp - Manual Execution Verification ----------------===//
//
// This file simulates what the compiler WOULD do.
// It manually builds a "Plan" and runs it to verify the Runtime stack.
//
//===----------------------------------------------------------------------===//

#include "Runtime/Core/HostContext.h"
#include "Runtime/Executor/ExecutionEngine.h"
#include "Runtime/Kernels/KernelRegistration.h"
#include <iostream>
#include <vector>
#include <cassert>

// Mock Tensor for the test (since we might not link against full TensorLib)
// In a real verification, we'd link against TensorLib.
struct MockTensor {
    float val;
    MockTensor(float v) : val(v) {}
    MockTensor& operator+(const MockTensor& other) {
        val += other.val;
        return *this;
    }
};

// We need to override the AddWrapper for this TEST to use MockTensor
// because HostKernels.cpp uses the real (or different mock) Tensor.
// So we'll register a "test.add" kernel here.
using namespace nova::runtime;

AsyncValue* TestAddWrapper(const std::vector<AsyncValue*>& args, HostContext* host) {
    auto* lhs_av = dynamic_cast<ConcreteAsyncValue<void*>*>(args[0]);
    auto* rhs_av = dynamic_cast<ConcreteAsyncValue<void*>*>(args[1]);
    
    MockTensor* lhs = static_cast<MockTensor*>(lhs_av->get());
    MockTensor* rhs = static_cast<MockTensor*>(rhs_av->get());
    
    std::cout << "  [KERNEL] Executing test.add: " << lhs->val << " + " << rhs->val << "\n";
    
    MockTensor* result = new MockTensor(lhs->val + rhs->val);
    return host->MakeAvailableAsyncValue<void*>(result);
}

// --- JIT Function Mock ---
// Signature: void* func(void** args)
// args[0] = input (MockTensor*)
// Returns (MockTensor*)
extern "C" void* jit_mock_mul_by_2(void** args) {
    MockTensor* input = static_cast<MockTensor*>(args[0]);
    std::cout << "  [JIT] Executing jit_mock_mul_by_2: " << input->val << " * 2\n";
    return new MockTensor(input->val * 2.0f);
}

int main() {
    std::cout << "=== Verification: Runtime Flow (Hybrid) ===\n";

    // 1. Setup Host
    HostContext host(4);
    KernelRegistry::Instance().RegisterKernel("test.add", Device::CPU, TestAddWrapper);

    // 2. Inputs
    MockTensor* inputA = new MockTensor(10.0f);
    MockTensor* inputB = new MockTensor(20.0f);
    std::vector<void*> inputs = { inputA, inputB };

    // 3. Plan:
    // Task 0: Add(In0, In1) -> 30.0 (Library)
    // Task 1: JIT_Mul2(Task0) -> 60.0 (JIT)
    
    RuntimeExecutionPlan plan;
    plan.output_task_id = 1;

    // Task 0: Library Add
    AsyncTask task0;
    task0.task_id = 0;
    task0.op_name = "test.add";
    task0.device = Device::CPU;
    task0.args = { ArgInput{0}, ArgInput{1} };
    plan.tasks.push_back(task0);

    // Task 1: JIT Mul
    AsyncTask task1;
    task1.task_id = 1;
    task1.op_name = "jit.generated"; // ignored by JIT launcher
    task1.device = Device::CPU;
    task1.dependencies = { 0 };       // Wait for Task 0
    task1.args = { ArgSlot{0} };      // Use result of Task 0
    
    // Casting function pointer to void*
    task1.jit_function = reinterpret_cast<void*>(&jit_mock_mul_by_2);
    
    plan.tasks.push_back(task1);

    // 4. Execute
    ExecutionEngine engine(&host);
    AsyncValue* result_av = engine.Execute(plan, inputs, {});
    result_av->Await();

    // 5. Check
    if (result_av->IsError()) {
        std::cerr << "FAILED: " << result_av->GetError() << "\n";
        return 1;
    }

    auto* concrete_result = dynamic_cast<ConcreteAsyncValue<void*>*>(result_av);
    MockTensor* res = static_cast<MockTensor*>(concrete_result->get());
    
    std::cout << "Result: " << res->val << "\n";
    
    if (res->val == 60.0f) {
        std::cout << "SUCCESS! ( (10+20) * 2 = 60 )\n";
    } else {
        std::cerr << "FAILURE: Expected 60.0, got " << res->val << "\n";
        return 1;
    }

    // Cleanup
    delete inputA;
    delete inputB;
    delete res;
    // Note: AsyncValues are ref-counted or managed by HostContext in real impl. 
    // Here we leak small amount for simple test.

    return 0;
}
