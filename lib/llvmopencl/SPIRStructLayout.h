// Header for preserving SPIR struct layouts across a datalayout change.
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

#ifndef POCL_SPIR_STRUCT_LAYOUT_H
#define POCL_SPIR_STRUCT_LAYOUT_H

#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Module.h>

#include <string>

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

namespace pocl {

// Rewrites the module, which still has the SPIR datalayout, so that its
// memory layout stays the same once the datalayout is replaced by
// TargetDL. Returns false and explains why in Log if the module uses a
// construct that can't be preserved.
bool preserveSPIRStructLayouts(llvm::Module &M,
                               const llvm::DataLayout &TargetDL,
                               std::string &Log);

} // namespace pocl

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#endif
