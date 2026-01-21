#ifndef NOVA_NOVATRAITS_H
#define NOVA_NOVATRAITS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace OpTrait {

/// This trait verifies that no operands have complex element types.
template <typename ConcreteType>
class NoComplexOperands : public TraitBase<ConcreteType, NoComplexOperands> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    for (Value operand : op->getOperands()) {
      Type type = operand.getType();
      // Check if the type itself is complex
      if (isa<ComplexType>(type)) {
        return op->emitOpError("From Trait:operand cannot have complex type");
      }
      // Check if it's a tensor with complex elements
      if (auto shapedType = dyn_cast<ShapedType>(type)) {
        if (isa<ComplexType>(shapedType.getElementType())) {
          return op->emitOpError("From Trait:operand cannot have complex element type");
        }
      }
    }
    return success();
  }
};
template <typename ConcreteType>
class NoIntegerOperands : public TraitBase<ConcreteType, NoIntegerOperands> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    for (Value operand : op->getOperands()) {
      Type type = operand.getType();
      // Check if the type itself is complex
      if (isa<IntegerType>(type)) {
        return op->emitOpError("From Trait:operand cannot have integer type");
      }
      // Check if it's a tensor with complex elements
      if (auto shapedType = dyn_cast<ShapedType>(type)) {
        if (isa<IntegerType>(shapedType.getElementType())) {
          return op->emitOpError("From Trait:operand cannot have integer element type");
        }
      }
    }
    return success();
  }
};
}
} 

#endif