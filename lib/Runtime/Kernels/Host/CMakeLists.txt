add_library(NovaRuntimeKernelsHost OBJECT
  HostKernels.cpp
)

target_include_directories(NovaRuntimeKernelsHost PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)

# If TensorLib headers are available, add include path
# target_include_directories(NovaRuntimeKernelsHost PRIVATE /path/to/tensor/include)
# For now, we rely on the parent project to set HAS_TENSORLIB and include paths if needed.
