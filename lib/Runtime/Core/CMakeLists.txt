add_library(NovaRuntimeCore
  AsyncValue.cpp
  HostContext.cpp
  MemoryAllocator.cpp
  ErrorHandling.cpp
)

target_include_directories(NovaRuntimeCore PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Link against pthread for std::thread/mutex if needed (usually handled by compiler/cmake automatically or Threads package)
find_package(Threads REQUIRED)
target_link_libraries(NovaRuntimeCore PUBLIC Threads::Threads)
