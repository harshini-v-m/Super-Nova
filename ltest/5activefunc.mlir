// Test file for activation functions
// This file demonstrates all activation operations in nova dialect

module {
  func.func @activation_functions(
    %arg0: tensor<4x8xf32, #nova.device<"1">>
  ) -> tensor<4x8xf32, #nova.device<"1">> {
    
    %relu = nova.relu %arg0 : tensor<4x8xf32, #nova.device<"1">>
    %sigmoid = nova.sigmoid %relu : tensor<4x8xf32, #nova.device<"1">>
    %softmax = nova.softmax %sigmoid : tensor<4x8xf32, #nova.device<"1">>
    %gelu = nova.gelu %softmax : tensor<4x8xf32, #nova.device<"1">>
    
    return %gelu : tensor<4x8xf32, #nova.device<"1">>
  }
}