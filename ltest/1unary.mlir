// Test file for unary operations: arithmetic, exponent, and trigonometry
// This file contains 3 functions demonstrating unary operations

module {
  // Function 1: Unary Arithmetic Operations
  func.func @unary_arithmetic(
    %arg0: tensor<4x8xf32, #nova.device<"1">>
  ) -> tensor<4x8xi1, #nova.device<"1">> {
    
    %square = nova.square %arg0 : tensor<4x8xf32, #nova.device<"1">>
    %sqrt   = nova.sqrt   %square : tensor<4x8xf32, #nova.device<"1">>
    %neg    = nova.neg    %sqrt : tensor<4x8xf32, #nova.device<"1">>
    %abs    = nova.abs    %neg : tensor<4x8xf32, #nova.device<"1">>
    %sign   = nova.sign   %abs : tensor<4x8xf32, #nova.device<"1">>
    %recip  = nova.reciprocal %sign : tensor<4x8xf32, #nova.device<"1">>
    %not    = "nova.not"(%recip) : (tensor<4x8xf32, #nova.device<"1">>) -> tensor<4x8xi1, #nova.device<"1">>
    
    return %not : tensor<4x8xi1, #nova.device<"1">>
  }

  // Function 2: Unary Exponent/Logarithm Operations
  func.func @unary_exponent(
    %arg0: tensor<4x8xf32, #nova.device<"1">>
  ) -> tensor<4x8xf32, #nova.device<"1">> {
    
    %exp   = nova.exp   %arg0 : tensor<4x8xf32, #nova.device<"1">>
    %exp2  = nova.exp2  %exp : tensor<4x8xf32, #nova.device<"1">>
    %log   = nova.log   %exp2 : tensor<4x8xf32, #nova.device<"1">>
    %log2  = nova.log2  %log : tensor<4x8xf32, #nova.device<"1">>
    %log10 = nova.log10 %log2 : tensor<4x8xf32, #nova.device<"1">>
    
    return %log10 : tensor<4x8xf32, #nova.device<"1">>
  }

  // Function 3: Unary Trigonometry Operations
  func.func @unary_trigonometry(
    %arg0: tensor<4x8xf32, #nova.device<"1">>
  ) -> tensor<4x8xf32, #nova.device<"1">> {
    
    // Basic trigonometric functions
    %sin  = nova.sin  %arg0 : tensor<4x8xf32, #nova.device<"1">>
    %cos  = nova.cos  %sin : tensor<4x8xf32, #nova.device<"1">>
    %tan  = nova.tan  %cos : tensor<4x8xf32, #nova.device<"1">>
    
    // Inverse trigonometric functions
    %asin = nova.asin %tan : tensor<4x8xf32, #nova.device<"1">>
    %acos = nova.acos %asin : tensor<4x8xf32, #nova.device<"1">>
    %atan = nova.atan %acos : tensor<4x8xf32, #nova.device<"1">>
    
    // Hyperbolic functions
    %sinh = nova.sinh %atan : tensor<4x8xf32, #nova.device<"1">>
    %cosh = nova.cosh %sinh : tensor<4x8xf32, #nova.device<"1">>
    %tanh = nova.tanh %cosh : tensor<4x8xf32, #nova.device<"1">>
    
    // Inverse hyperbolic functions
    %asinh = nova.asinh %tanh : tensor<4x8xf32, #nova.device<"1">>
    %acosh = nova.acosh %asinh : tensor<4x8xf32, #nova.device<"1">>
    %atanh = nova.atanh %acosh : tensor<4x8xf32, #nova.device<"1">>
    
    return %atanh : tensor<4x8xf32, #nova.device<"1">>
  }
}