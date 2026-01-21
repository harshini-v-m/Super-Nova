add_library(NovaRuntimeExecutor
  ExecutionEngine.cpp
  KernelRegistry.cpp
  DependencyGraph.cpp
)

target_include_directories(NovaRuntimeExecutor PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(NovaRuntimeExecutor PUBLIC
  NovaRuntimeCore
)
