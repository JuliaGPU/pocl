// Preserves the memory layout of SPIR struct types when a module is
// retargeted to the host datalayout.
//
// SPIR modules (e.g. from SPIR-V) carry the SPIR datalayout, which follows
// OpenCL C: long and double are 8-byte aligned. Linking them with the
// kernel library replaces that datalayout with the host's, where these
// types can be less aligned (4 bytes on i386). Struct types without explicit
// padding would then silently change their field offsets, sizes and array
// strides. Clang avoids this for OpenCL C by emitting explicitly padded
// structs; this does the same for SPIR modules.
//
// Every struct type whose layout differs between the two datalayouts is
// replaced, wherever it describes memory, by a packed struct with explicit
// [N x i8] padding that has the SPIR offsets and size under any datalayout.
// That covers GEPs (with renumbered field indices), allocas, globals and
// their initializers, and the byval/sret/byref/elementtype attributes.
// Aggregate SSA values have no layout, so they keep their original type;
// aggregate loads and stores of such types are split into per-element
// accesses at the SPIR offsets instead.
//
// Copyright (c) 2026 Tim Besard
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#include "config.h"

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wunused-parameter")
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Operator.h>
POP_COMPILER_DIAGS

#include "SPIRStructLayout.h"

using namespace llvm;

namespace {

class SPIRLayoutPreserver {
public:
  SPIRLayoutPreserver(Module &M, const DataLayout &TargetDL)
      : M(M), Ctx(M.getContext()), SPIRDL(M.getDataLayout()),
        TargetDL(TargetDL) {}

  bool run(std::string &Log);

private:
  Module &M;
  LLVMContext &Ctx;
  const DataLayout &SPIRDL;
  const DataLayout &TargetDL;

  // The replacement of every type looked at, or the type itself if its
  // layout is the same under both datalayouts.
  DenseMap<Type *, Type *> Replacements;
  // For each replaced struct, the index of each original field in the
  // replacement.
  DenseMap<StructType *, SmallVector<unsigned, 8>> FieldIndices;
  DenseMap<Constant *, Constant *> MappedConstants;
  std::string Error;

  Type *replace(Type *T);
  Type *replaceStruct(StructType *ST);
  template <typename RangeT>
  SmallVector<Value *, 8> mapIndices(Type *SrcTy, RangeT Indices);
  Constant *mapConstant(Constant *C);
  Constant *mapValueConstant(Constant *C);
  AttributeList mapTypeAttributes(AttributeList Attrs);
  void replaceGlobals();
  void rewriteInstructions(Function &F);
  Value *splitLoad(IRBuilder<> &B, Type *Ty, Value *Ptr, uint64_t Offset,
                   Align A);
  void splitStore(IRBuilder<> &B, Value *V, Value *Ptr, uint64_t Offset,
                  Align A);
};

Type *SPIRLayoutPreserver::replace(Type *T) {
  auto It = Replacements.find(T);
  if (It != Replacements.end())
    return It->second;

  Type *R = T;
  if (auto *AT = dyn_cast<ArrayType>(T)) {
    Type *Elt = replace(AT->getElementType());
    if (Elt != AT->getElementType())
      R = ArrayType::get(Elt, AT->getNumElements());
  } else if (auto *ST = dyn_cast<StructType>(T)) {
    if (!ST->isOpaque())
      R = replaceStruct(ST);
  }
  Replacements[T] = R;
  return R;
}

Type *SPIRLayoutPreserver::replaceStruct(StructType *ST) {
  SmallVector<Type *, 8> Elts;
  bool EltReplaced = false;
  for (Type *Elt : ST->elements()) {
    Elts.push_back(replace(Elt));
    EltReplaced |= Elts.back() != Elt;
  }

  const StructLayout *SPIRLayout = SPIRDL.getStructLayout(ST);
  if (!EltReplaced) {
    const StructLayout *TargetLayout = TargetDL.getStructLayout(ST);
    bool Same = SPIRLayout->getSizeInBytes() == TargetLayout->getSizeInBytes();
    for (unsigned I = 0; Same && I < ST->getNumElements(); ++I)
      Same =
          SPIRLayout->getElementOffset(I) == TargetLayout->getElementOffset(I);
    if (Same)
      return ST;
  }

  // Packed, so that the padding alone determines the offsets.
  SmallVector<Type *, 8> Fields;
  SmallVector<unsigned, 8> Indices;
  uint64_t Offset = 0;
  auto padTo = [&](uint64_t To) {
    if (To > Offset)
      Fields.push_back(ArrayType::get(Type::getInt8Ty(Ctx), To - Offset));
    Offset = To;
  };
  for (unsigned I = 0; I < Elts.size(); ++I) {
    padTo(SPIRLayout->getElementOffset(I));
    Indices.push_back(Fields.size());
    Fields.push_back(Elts[I]);
    Offset += TargetDL.getTypeAllocSize(Elts[I]).getFixedValue();
  }
  padTo(SPIRLayout->getSizeInBytes());

  StructType *R =
      ST->hasName()
          ? StructType::create(Ctx, Fields, (ST->getName() + ".spir").str(),
                               /*isPacked=*/true)
          : StructType::get(Ctx, Fields, /*isPacked=*/true);

  // Only fails if an element's size differs between the datalayouts.
  const StructLayout *NewLayout = TargetDL.getStructLayout(R);
  bool Same = NewLayout->getSizeInBytes() == SPIRLayout->getSizeInBytes();
  for (unsigned I = 0; Same && I < Elts.size(); ++I)
    Same = NewLayout->getElementOffset(Indices[I]) ==
           SPIRLayout->getElementOffset(I);
  if (!Same && Error.empty()) {
    raw_string_ostream OS(Error);
    OS << "cannot preserve the SPIR layout of " << *ST << "\n";
  }

  FieldIndices[ST] = std::move(Indices);
  return R;
}

// Maps the indices of a GEP over SrcTy to the ones over its replacement.
template <typename RangeT>
SmallVector<Value *, 8> SPIRLayoutPreserver::mapIndices(Type *SrcTy,
                                                        RangeT Indices) {
  SmallVector<Value *, 8> Result;
  Type *Ty = nullptr;
  for (Value *Idx : Indices) {
    if (Ty == nullptr) {
      // The first index steps over whole objects, whose size is preserved.
      Result.push_back(Idx);
      Ty = SrcTy;
      continue;
    }
    if (auto *ST = dyn_cast<StructType>(Ty)) {
      auto *C = cast<Constant>(Idx);
      if (C->getType()->isVectorTy())
        C = C->getSplatValue();
      unsigned Field = cast<ConstantInt>(C)->getZExtValue();
      auto It = FieldIndices.find(ST);
      Result.push_back(
          It == FieldIndices.end()
              ? Idx
              : ConstantInt::get(Idx->getType(), It->second[Field]));
      Ty = ST->getElementType(Field);
    } else {
      Result.push_back(Idx);
      Ty = Ty->isArrayTy() ? Ty->getArrayElementType()
                           : cast<VectorType>(Ty)->getElementType();
    }
  }
  return Result;
}

Constant *SPIRLayoutPreserver::mapConstant(Constant *C) {
  auto It = MappedConstants.find(C);
  if (It != MappedConstants.end())
    return It->second;

  Constant *R = C;
  Type *Ty = C->getType();
  Type *NewTy = replace(Ty);
  if (NewTy != Ty) {
    if (isa<PoisonValue>(C))
      R = PoisonValue::get(NewTy);
    else if (isa<UndefValue>(C))
      R = UndefValue::get(NewTy);
    else if (C->isNullValue())
      R = Constant::getNullValue(NewTy);
    else if (auto *ST = dyn_cast<StructType>(Ty)) {
      auto *NewST = cast<StructType>(NewTy);
      SmallVector<Constant *, 8> Fields;
      for (Type *FieldTy : NewST->elements())
        Fields.push_back(Constant::getNullValue(FieldTy));
      const auto &Indices = FieldIndices[ST];
      for (unsigned I = 0; I < ST->getNumElements(); ++I)
        Fields[Indices[I]] = mapConstant(C->getAggregateElement(I));
      R = ConstantStruct::get(NewST, Fields);
    } else {
      auto *AT = cast<ArrayType>(Ty);
      SmallVector<Constant *, 8> Elts;
      for (unsigned I = 0; I < AT->getNumElements(); ++I)
        Elts.push_back(mapConstant(C->getAggregateElement(I)));
      R = ConstantArray::get(cast<ArrayType>(NewTy), Elts);
    }
  } else if (isa<ConstantExpr>(C) || isa<ConstantAggregate>(C)) {
    SmallVector<Constant *, 8> Ops;
    bool Changed = false;
    for (Value *Op : C->operands()) {
      Ops.push_back(mapConstant(cast<Constant>(Op)));
      Changed |= Ops.back() != Op;
    }
    Type *NewSrcTy = nullptr;
    if (auto *GEP = dyn_cast<GEPOperator>(C)) {
      Type *SrcTy = GEP->getSourceElementType();
      if (replace(SrcTy) != SrcTy) {
        NewSrcTy = replace(SrcTy);
        auto Indices = mapIndices(SrcTy, GEP->indices());
        for (unsigned I = 0; I < Indices.size(); ++I)
          Ops[I + 1] = cast<Constant>(Indices[I]);
      }
    }
    if (NewSrcTy) {
      // getWithOperands would ignore a new source type if nothing else
      // changed, e.g. for GEPs that only index arrays of a replaced type.
      auto *GEP = cast<GEPOperator>(C);
      ArrayRef<Constant *> Indices = ArrayRef(Ops).drop_front();
#if LLVM_MAJOR < 19
      R = ConstantExpr::getGetElementPtr(
          NewSrcTy, Ops[0], Indices, GEP->isInBounds(), GEP->getInRangeIndex());
#else
      R = ConstantExpr::getGetElementPtr(
          NewSrcTy, Ops[0], Indices, GEP->getNoWrapFlags(), GEP->getInRange());
#endif
    } else if (Changed) {
      if (auto *CE = dyn_cast<ConstantExpr>(C))
        R = CE->getWithOperands(Ops, Ty);
      else if (auto *ST = dyn_cast<StructType>(Ty))
        R = ConstantStruct::get(ST, Ops);
      else if (auto *AT = dyn_cast<ArrayType>(Ty))
        R = ConstantArray::get(AT, Ops);
      else
        R = ConstantVector::get(Ops);
    }
  }
  MappedConstants[C] = R;
  return R;
}

// Like mapConstant, but for an SSA operand: aggregates keep their type.
Constant *SPIRLayoutPreserver::mapValueConstant(Constant *C) {
  if (replace(C->getType()) == C->getType())
    return mapConstant(C);
  if (!isa<ConstantAggregate>(C))
    return C;
  SmallVector<Constant *, 8> Ops;
  bool Changed = false;
  for (Value *Op : C->operands()) {
    Ops.push_back(mapValueConstant(cast<Constant>(Op)));
    Changed |= Ops.back() != Op;
  }
  if (!Changed)
    return C;
  if (auto *ST = dyn_cast<StructType>(C->getType()))
    return ConstantStruct::get(ST, Ops);
  return ConstantArray::get(cast<ArrayType>(C->getType()), Ops);
}

AttributeList SPIRLayoutPreserver::mapTypeAttributes(AttributeList Attrs) {
  static const Attribute::AttrKind Kinds[] = {
      Attribute::ByVal,    Attribute::StructRet,    Attribute::ByRef,
      Attribute::InAlloca, Attribute::Preallocated, Attribute::ElementType};
  // Not only parameters: llvm.call.preallocated.arg calls carry
  // preallocated as a function attribute.
  AttributeList Orig = Attrs;
  for (unsigned Index : Orig.indexes()) {
    for (Attribute::AttrKind Kind : Kinds) {
      Attribute A = Orig.getAttributeAtIndex(Index, Kind);
      if (!A.isValid())
        continue;
      Type *Ty = A.getValueAsType();
      Type *NewTy = replace(Ty);
      if (NewTy != Ty)
        Attrs = Attrs.replaceAttributeTypeAtIndex(Ctx, Index, Kind, NewTy);

      // Without an explicit alignment, the pointee is aligned like its type,
      // which the replacement (packed) or the target datalayout may lower.
      if (Kind != Attribute::ElementType &&
          Index != AttributeList::FunctionIndex &&
          Index >= AttributeList::FirstArgIndex) {
        unsigned ArgNo = Index - AttributeList::FirstArgIndex;
        Align SPIRAlign = SPIRDL.getABITypeAlign(Ty);
        if (!Attrs.getParamAlignment(ArgNo) &&
            TargetDL.getABITypeAlign(NewTy) != SPIRAlign)
          Attrs = Attrs.addParamAttribute(
              Ctx, ArgNo, Attribute::getWithAlignment(Ctx, SPIRAlign));
      }
    }
  }
  return Attrs;
}

void SPIRLayoutPreserver::replaceGlobals() {
  SmallVector<std::pair<GlobalVariable *, GlobalVariable *>, 8> Replaced;
  for (GlobalVariable &GV : M.globals()) {
    Type *Ty = GV.getValueType();
    Type *NewTy = replace(Ty);
    if (NewTy == Ty)
      continue;
    auto *NewGV =
        new GlobalVariable(M, NewTy, GV.isConstant(), GV.getLinkage(), nullptr,
                           "", &GV, GV.getThreadLocalMode(),
                           GV.getAddressSpace(), GV.isExternallyInitialized());
    NewGV->copyAttributesFrom(&GV);
    NewGV->setComdat(GV.getComdat());
    NewGV->copyMetadata(&GV, 0);
    NewGV->setAlignment(GV.getAlign() ? *GV.getAlign()
                                      : SPIRDL.getPreferredAlign(&GV));
    Replaced.push_back({&GV, NewGV});
  }

  // Pointers are opaque, so the replacements can take over every use, also
  // in the initializers of other globals, before those are mapped below.
  for (auto [GV, NewGV] : Replaced)
    GV->replaceAllUsesWith(NewGV);
  for (auto [GV, NewGV] : Replaced) {
    if (GV->hasInitializer())
      NewGV->setInitializer(mapConstant(GV->getInitializer()));
    NewGV->takeName(GV);
    GV->eraseFromParent();
  }

  for (GlobalVariable &GV : M.globals()) {
    if (!GV.getAlign() &&
        SPIRDL.getPreferredAlign(&GV) != TargetDL.getPreferredAlign(&GV))
      GV.setAlignment(SPIRDL.getPreferredAlign(&GV));
    if (!GV.hasInitializer())
      continue;
    Constant *Init = mapConstant(GV.getInitializer());
    if (Init != GV.getInitializer())
      GV.setInitializer(Init);
  }
  for (GlobalAlias &GA : M.aliases()) {
    Constant *Aliasee = mapConstant(GA.getAliasee());
    if (Aliasee != GA.getAliasee())
      GA.setAliasee(Aliasee);
  }
}

Value *SPIRLayoutPreserver::splitLoad(IRBuilder<> &B, Type *Ty, Value *Ptr,
                                      uint64_t Offset, Align A) {
  if (auto *ST = dyn_cast<StructType>(Ty)) {
    const StructLayout *Layout = SPIRDL.getStructLayout(ST);
    Value *V = PoisonValue::get(ST);
    for (unsigned I = 0; I < ST->getNumElements(); ++I)
      V = B.CreateInsertValue(V,
                              splitLoad(B, ST->getElementType(I), Ptr,
                                        Offset + Layout->getElementOffset(I),
                                        A),
                              I);
    return V;
  }
  if (auto *AT = dyn_cast<ArrayType>(Ty)) {
    uint64_t Stride =
        SPIRDL.getTypeAllocSize(AT->getElementType()).getFixedValue();
    Value *V = PoisonValue::get(AT);
    for (unsigned I = 0; I < AT->getNumElements(); ++I)
      V = B.CreateInsertValue(
          V, splitLoad(B, AT->getElementType(), Ptr, Offset + I * Stride, A),
          I);
    return V;
  }
  Value *P =
      Offset ? B.CreateConstInBoundsGEP1_64(B.getInt8Ty(), Ptr, Offset) : Ptr;
  return B.CreateAlignedLoad(Ty, P, commonAlignment(A, Offset));
}

void SPIRLayoutPreserver::splitStore(IRBuilder<> &B, Value *V, Value *Ptr,
                                     uint64_t Offset, Align A) {
  Type *Ty = V->getType();
  if (auto *ST = dyn_cast<StructType>(Ty)) {
    const StructLayout *Layout = SPIRDL.getStructLayout(ST);
    for (unsigned I = 0; I < ST->getNumElements(); ++I)
      splitStore(B, B.CreateExtractValue(V, I), Ptr,
                 Offset + Layout->getElementOffset(I), A);
    return;
  }
  if (auto *AT = dyn_cast<ArrayType>(Ty)) {
    uint64_t Stride =
        SPIRDL.getTypeAllocSize(AT->getElementType()).getFixedValue();
    for (unsigned I = 0; I < AT->getNumElements(); ++I)
      splitStore(B, B.CreateExtractValue(V, I), Ptr, Offset + I * Stride, A);
    return;
  }
  Value *P =
      Offset ? B.CreateConstInBoundsGEP1_64(B.getInt8Ty(), Ptr, Offset) : Ptr;
  B.CreateAlignedStore(V, P, commonAlignment(A, Offset));
}

void SPIRLayoutPreserver::rewriteInstructions(Function &F) {
  SmallVector<Instruction *, 8> ToSplit;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      for (Use &U : I.operands()) {
        auto *C = dyn_cast<Constant>(U.get());
        if (C == nullptr || isa<GlobalValue>(C))
          continue;
        Constant *NewC = mapValueConstant(C);
        if (NewC != C)
          U.set(NewC);
      }

      if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
        Type *SrcTy = GEP->getSourceElementType();
        Type *NewSrcTy = replace(SrcTy);
        if (NewSrcTy == SrcTy)
          continue;
        auto Indices = mapIndices(SrcTy, GEP->indices());
        for (unsigned Idx = 0; Idx < Indices.size(); ++Idx)
          GEP->setOperand(Idx + 1, Indices[Idx]);
        GEP->setSourceElementType(NewSrcTy);
        GEP->setResultElementType(replace(GEP->getResultElementType()));
      } else if (auto *AI = dyn_cast<AllocaInst>(&I)) {
        AI->setAllocatedType(replace(AI->getAllocatedType()));
      } else if (auto *CB = dyn_cast<CallBase>(&I)) {
        CB->setAttributes(mapTypeAttributes(CB->getAttributes()));
      } else if (auto *LI = dyn_cast<LoadInst>(&I)) {
        if (replace(LI->getType()) != LI->getType())
          ToSplit.push_back(LI);
      } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
        Type *Ty = SI->getValueOperand()->getType();
        if (replace(Ty) != Ty)
          ToSplit.push_back(SI);
      }
    }
  }

  for (Instruction *I : ToSplit) {
    IRBuilder<> B(I);
    if (auto *LI = dyn_cast<LoadInst>(I)) {
      if (LI->isVolatile()) {
        Error = "volatile aggregate loads are not supported\n";
        return;
      }
      Value *V = splitLoad(B, LI->getType(), LI->getPointerOperand(), 0,
                           LI->getAlign());
      V->takeName(LI);
      LI->replaceAllUsesWith(V);
    } else {
      auto *SI = cast<StoreInst>(I);
      if (SI->isVolatile()) {
        Error = "volatile aggregate stores are not supported\n";
        return;
      }
      splitStore(B, SI->getValueOperand(), SI->getPointerOperand(), 0,
                 SI->getAlign());
    }
    I->eraseFromParent();
  }
}

bool SPIRLayoutPreserver::run(std::string &Log) {
  // Kernel arguments are passed in memory, which the launcher would read
  // with the target layout.
  for (Function &F : M) {
    if (F.getCallingConv() != CallingConv::SPIR_KERNEL)
      continue;
    for (Argument &A : F.args()) {
      if (!A.hasByValAttr() && replace(A.getType()) != A.getType()) {
        Log = "kernel " + F.getName().str() +
              " takes an aggregate argument by value, which is only "
              "supported through a byval pointer\n";
        return false;
      }
    }
  }

  replaceGlobals();
  for (Function &F : M) {
    F.setAttributes(mapTypeAttributes(F.getAttributes()));
    rewriteInstructions(F);
  }

  if (!Error.empty()) {
    Log = Error;
    return false;
  }
  return true;
}

} // namespace

namespace pocl {

bool preserveSPIRStructLayouts(Module &M, const DataLayout &TargetDL,
                               std::string &Log) {
  return SPIRLayoutPreserver(M, TargetDL).run(Log);
}

} // namespace pocl
