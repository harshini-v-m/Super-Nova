module {
  func.func @test_cpu(%arg0: tensor<8x8xf32, #nova.device<"1">>, %arg1: tensor<8x8xf32, #nova.device<"1">>) 
  -> tensor<8x8xf32, #nova.device<"0">> {
    %0 = nova.add %arg0, %arg1 : tensor<8x8xf32, #nova.device<"1">>, tensor<8x8xf32, #nova.device<"1">>
    %1 = nova.matmul %0, %arg1 : tensor<8x8xf32, #nova.device<"1">>, tensor<8x8xf32, #nova.device<"1">>
    %2=nova.to_device %1 : tensor<8x8xf32, #nova.device<"1">> -> tensor<8x8xf32, #nova.device<"0">>
    return %2 : tensor<8x8xf32, #nova.device<"0">>
  }
}
