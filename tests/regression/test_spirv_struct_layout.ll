; Aggregate SSA values, aggregate loads and stores, and typed constant GEPs,
; which OpenCL C compilers don't emit but other SPIR-V producers do. Linked into
; test_spirv_struct_layout{32,64}.spv, see test_spirv_struct_layout.cl.
;
; P: a at 0, t at 8, l at 24, size 32

%T = type { double, i32 }
%P = type { i32, %T, i64 }

@cp = internal addrspace(2) constant %P { i32 10, %T { double 1.5, i32 11 }, i64 12 }, align 8

; a constant GEP that only steps over an array of T, whose stride changes
@tarr = internal addrspace(2) constant [2 x %T] [%T { double 1.0, i32 1 }, %T { double 2.0, i32 2 }], align 8
@tp = internal addrspace(2) constant ptr addrspace(2) getelementptr ([2 x %T], ptr addrspace(2) @tarr, i32 0, i32 1), align 4

define internal spir_func %P @bump(%P %v) {
  %a = extractvalue %P %v, 0
  %a1 = add i32 %a, 1
  %v1 = insertvalue %P %v, i32 %a1, 0
  %d = extractvalue %P %v1, 1, 0
  %d1 = fmul double %d, 2.0
  %v2 = insertvalue %P %v1, double %d1, 1, 0
  %l = extractvalue %P %v2, 2
  %l1 = add i64 %l, 100
  %v3 = insertvalue %P %v2, i64 %l1, 2
  ret %P %v3
}

define spir_kernel void @aggregate(ptr addrspace(1) align 8 %io) {
  %v = load %P, ptr addrspace(1) %io, align 8
  %w = call spir_func %P @bump(%P %v)
  store %P %w, ptr addrspace(1) %io, align 8
  %c = load %P, ptr addrspace(2) @cp, align 8
  %io1 = getelementptr inbounds %P, ptr addrspace(1) %io, i32 1
  store %P %c, ptr addrspace(1) %io1, align 8
  %io2 = getelementptr inbounds %P, ptr addrspace(1) %io, i32 2
  store %P { i32 20, %T { double 2.5, i32 21 }, i64 22 }, ptr addrspace(1) %io2, align 8
  ret void
}

define spir_kernel void @constant_gep(ptr addrspace(1) align 4 %out) {
  %p = load ptr addrspace(2), ptr addrspace(2) @tp, align 4
  %f = getelementptr inbounds %T, ptr addrspace(2) %p, i32 0, i32 1
  %v = load i32, ptr addrspace(2) %f, align 8
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}
