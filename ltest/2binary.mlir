// Test file for binary operations: arithmetic and boolean
// This file contains 2 functions demonstrating binary operations

module {
  // Function 1: Binary Arithmetic Operations
  func.func @binary_arithmetic(
    %arg0: tensor<4x8xf32, #nova.device<"1">>,
    %arg1: tensor<4x8xf32, #nova.device<"1">>
  ) -> tensor<4x8xf32, #nova.device<"1">> {
    
    %add = nova.add %arg0, %arg1 : tensor<4x8xf32, #nova.device<"1">>, tensor<4x8xf32, #nova.device<"1">>
    %sub = nova.sub %add, %arg1 : tensor<4x8xf32, #nova.device<"1">>, tensor<4x8xf32, #nova.device<"1">>
    %mul = nova.mul %sub, %arg1 : tensor<4x8xf32, #nova.device<"1">>, tensor<4x8xf32, #nova.device<"1">>
    %div = nova.div %mul, %arg1 : tensor<4x8xf32, #nova.device<"1">>, tensor<4x8xf32, #nova.device<"1">>
    %mod = nova.mod %div, %arg1 : tensor<4x8xf32, #nova.device<"1">>, tensor<4x8xf32, #nova.device<"1">>
    %pow = nova.pow %mod, %arg1 : tensor<4x8xf32, #nova.device<"1">>, tensor<4x8xf32, #nova.device<"1">>
    %max = nova.max %pow, %arg1 : tensor<4x8xf32, #nova.device<"1">>, tensor<4x8xf32, #nova.device<"1">>
    %min = nova.min %max, %arg1 : tensor<4x8xf32, #nova.device<"1">>, tensor<4x8xf32, #nova.device<"1">>
    
    return %add : tensor<4x8xf32, #nova.device<"1">>
  }

  // Function 2: Binary Boolean Operations
  func.func @binary_boolean(
    %arg0: tensor<4x8xf32, #nova.device<"1">>,
    %arg1: tensor<4x8xf32, #nova.device<"1">>
  ) -> tensor<4x8xi1, #nova.device<"1">> {
    
    %and = "nova.and"(%arg0, %arg1) : (tensor<4x8xf32, #nova.device<"1">>, tensor<4x8xf32, #nova.device<"1">>) -> tensor<4x8xi1, #nova.device<"1">>
    %or  = "nova.or"(%and, %arg1) : (tensor<4x8xi1, #nova.device<"1">>, tensor<4x8xf32, #nova.device<"1">>) -> tensor<4x8xi1, #nova.device<"1">>
    %xor = "nova.xor"(%or,  %arg1) : (tensor<4x8xi1, #nova.device<"1">>, tensor<4x8xf32, #nova.device<"1">>) -> tensor<4x8xi1, #nova.device<"1">>
    
    return %xor : tensor<4x8xi1, #nova.device<"1">>
  }
}