cmake_minimum_required(VERSION 3.10)
find_package(Threads REQUIRED)

add_executable(RuntimeFlowTest RuntimeFlowTest.cpp)

target_include_directories(RuntimeFlowTest PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(RuntimeFlowTest PRIVATE
  NovaRuntimeCore
  NovaRuntimeExecutor
  NovaRuntimeKernels
  Threads::Threads
)
