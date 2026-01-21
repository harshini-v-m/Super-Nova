find_package(LLVM REQUIRED CONFIG)
add_mlir_tool(nova-translate
  nova-translate.cpp
)

target_link_libraries(nova-translate PRIVATE

  MLIRTargetLLVMIRExport
  MLIRSupport
  MLIRIR
  MLIRTargetLLVMIRExport
  MLIRLLVMDialect
  
  LLVMX86CodeGen
  LLVMX86Info
  LLVMX86Desc
  LLVMX86AsmParser
  LLVMSupport
  LLVMCore
  LLVMTarget
  LLVMMC
  LLVMAsmPrinter
  LLVMSelectionDAG 
  MLIRToLLVMIRTranslationRegistration
  MLIRLLVMToLLVMIRTranslation
)

target_include_directories(nova-translate PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${LLVM_INCLUDE_DIRS}
)
