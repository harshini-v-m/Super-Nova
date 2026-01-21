## Docker file

```
docker pull adwaid10/mlir-compiler
```
### Documentation : [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/AdwaidSuresh123/MlirCompiler)

----
**LLVM Version**:
```
21.1.6
```

**LLVM Commit Hash**:
```
git checkout a832a5222e489298337fbb5876f8dcaf072c5cca
```

**LLVM Build**:
```
mkdir build
cd build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=OFF \
   -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON
   -DCMAKE_C_COMPILER=clang \
   -DCMAKE_CXX_COMPILER=clang++ \
   -DLLVM_ENABLE_LLD=ON \
   -DLLVM_CCACHE_BUILD=ON \
```
