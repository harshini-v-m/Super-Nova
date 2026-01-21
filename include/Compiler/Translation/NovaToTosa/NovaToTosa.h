#ifndef MLIR_CONVERSION_NOVATOTOSA_H
#define MLIR_CONVERSION_NOVATOTOSA_H

#include "mlir/Pass/Pass.h"
#include <memory>
namespace mlir{
    class Pass;
    class RewritePatternSet;
    class TypeConverter;
    namespace nova{
        std::unique_ptr<Pass> createNovaToTosaLoweringPass();
        void registerNovaToTosaLoweringPass();
        void populateNovaToTosaConversionPatterns(RewritePatternSet &patterns);
        void populateNovaToTosaTemplatePatterns(RewritePatternSet &Patterns);

    }
}
#endif
//patterns
/*
nova.max       -> tosa.maximum (int,float)
nova.min       -> tosa.minimum(int,float)
nova.Transpose -> tosa.transpose(int,float)
nova.And       -> tosa.logicaland(int,float)
nova.or        -> tosa.logicalor(int,float)
nova.xor       -> tosa.logicalxor(int,float)
nova.not       -> tosa.logicalnot(int,float)
nova.sigmoid   -> tosa.sigmoid(int,float)
nova.neg       -> tosa.negate(int,float),linalg.generic(complex)
nova.sin       -> tosa.sin(int,float),linalg.generic(complex)
nova.cos       -> tosa.cos(int,float),linalg.generic(complex)
nova.tanh      -> tosa.tanh(int,float),linalg.generic(complex)
nova.reciprocal-> tosa.reciprocal
nova.log       -> tosa.log(int,float),linalg.generic(complex)
nova.exp       -> tosa.exp(int,float),linalg.generic(complex)
nova.abs       -> tosa.abs(int,float),linalg.generic(complex)
nova.gelu      -> tosa and nova named ops(int,float)
nova.relu      -> tosa and nova named ops(int,float)
reduce operations: ---

(int,float)
nova.add       ->tosa.add
nova.sub       ->tosa.sub
nova.mul       ->tosa.mul
nova.pow       ->tosa.pow
nova.sqrt       ->tosa.pow

**/