#include "Compiler/Dialect/nova/NovaDialect.h"
#include "Compiler/Dialect/nova/NovaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::nova;

//===----------------------------------------------------------------------===//
// Nova Dialect
//===----------------------------------------------------------------------===//

void NovaDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Compiler/Dialect/nova/NovaOps.cpp.inc"
  >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Compiler/Dialect/nova/NovaOpsAttributes.cpp.inc"
  >();
}

// Include generated dialect definitions
#include "Compiler/Dialect/nova/NovaOpsDialect.cpp.inc"

namespace llvm {
template <typename T, unsigned N>
hash_code hash_value(const SmallVector<T, N> &arg) {
  return hash_value(ArrayRef<T>(arg));
}
}

#define GET_ATTRDEF_CLASSES
#include "Compiler/Dialect/nova/NovaOpsAttributes.cpp.inc"
