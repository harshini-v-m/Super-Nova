add_library(NovaRuntimeKernelsCUDA OBJECT
  CudaKernels.cpp
)

target_include_directories(NovaRuntimeKernelsCUDA PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)
