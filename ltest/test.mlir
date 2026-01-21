module {
  func.func @main(%arg0: tensor<8x16xf32, #nova.device<"0">>,
  %arg1: tensor<1x16xf32, #nova.device<"0">>) -> 
  tensor<8x16xf32  , #nova.device<"0">> {
    %1 = nova.add %arg0 ,%arg1 : tensor<8x16xf32, #nova.device<"0">>,tensor<1x16xf32, #nova.device<"0">>
    return %1 : tensor<8x16xf32, #nova.device<"0">>
  }
}