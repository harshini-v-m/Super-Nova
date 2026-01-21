module {
  func.func @main(%arg0: tensor<8x16xf32, #nova.device<"0">>, %arg1: tensor<16x10xf32, #nova.device<"0">>, %arg2: tensor<1x10xf32, #nova.device<"0">>, %arg3: tensor<8x10xf32, #nova.device<"0">>) -> tensor<f32, #nova.device<"0">> {
    %0 = nova.matmul %arg0, %arg1 : tensor<8x16xf32, #nova.device<"0">>, tensor<16x10xf32, #nova.device<"0">>
    %1 = nova.add %0, %arg2 : tensor<8x10xf32, #nova.device<"0">>, tensor<1x10xf32, #nova.device<"0">>
    %2 = nova.bce %1, %arg3 : tensor<8x10xf32, #nova.device<"0">>, tensor<8x10xf32, #nova.device<"0">>
    return %2 : tensor<f32, #nova.device<"0">>
  }
}