add_subdirectory(Host)
add_subdirectory(CUDA)

# Collect all kernel objects into a single library
add_library(NovaRuntimeKernels
  OBJECT
  $<TARGET_OBJECTS:NovaRuntimeKernelsHost>
  $<TARGET_OBJECTS:NovaRuntimeKernelsCUDA>
)

target_include_directories(NovaRuntimeKernels PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Link against Core and Executor
target_link_libraries(NovaRuntimeKernels PUBLIC NovaRuntimeCore NovaRuntimeExecutor)
