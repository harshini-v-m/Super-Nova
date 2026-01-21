
#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

int main(int argc, char **argv) {
  // Register only the LLVM IR translation
  mlir::registerToLLVMIRTranslation();
  
  return failed(mlir::mlirTranslateMain(argc, argv, "Nova Translation Tool"));
}