// Test file for matrix operations and device transfers
// This includes matmul, transpose, and device transfer operations

module {
  // Matrix operations
  func.func @matrix_operations(
    %arg0: tensor<4x8xf32, #nova.device<"1">>,
    %arg1: tensor<8x16xf32, #nova.device<"1">>
  ) -> (tensor<4x16xf32, #nova.device<"1">>, tensor<8x4xf32, #nova.device<"1">>) {
    
    // Matrix multiplication
    %matmul = nova.matmul %arg0, %arg1 
      : tensor<4x8xf32, #nova.device<"1">>, tensor<8x16xf32, #nova.device<"1">>
    
    // Transpose
    %transpose = nova.transpose %arg0 axes1 = 0 axes2 = 1
      : tensor<4x8xf32, #nova.device<"1">>
    
    return %matmul, %transpose 
      : tensor<4x16xf32, #nova.device<"1">>, tensor<8x4xf32, #nova.device<"1">>
  }

  // Device transfer operations
  func.func @device_transfer(
    %arg0: tensor<4x8xf32, #nova.device<"0">>
  ) -> (tensor<4x8xf32, #nova.device<"1">>, tensor<4x8xf32, #nova.device<"0">>) {
    
    // Transfer from CPU (device 0) to GPU (device 1)
    %to_gpu = nova.to_device %arg0 
      : tensor<4x8xf32, #nova.device<"0">> -> tensor<4x8xf32, #nova.device<"1">>
    
    // Transfer from GPU back to CPU
    %to_cpu = nova.to_device %to_gpu 
      : tensor<4x8xf32, #nova.device<"1">> -> tensor<4x8xf32, #nova.device<"0">>
    
    return %to_gpu, %to_cpu 
      : tensor<4x8xf32, #nova.device<"1">>, tensor<4x8xf32, #nova.device<"0">>
  }
}