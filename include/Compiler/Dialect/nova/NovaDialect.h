#ifndef NOVA_DIALECT_H
#define NOVA_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "Compiler/Dialect/nova/NovaOpsDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "Compiler/Dialect/nova/NovaOpsAttributes.h.inc"


#endif // NOVA_DIALECT_H
