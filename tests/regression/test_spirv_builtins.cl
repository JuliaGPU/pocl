/* Kernels for test_spirv_builtins.c, calling builtins whose SPIR wrappers
   depend on the address bits or the calling convention of the host. Built
   to test_spirv_builtins{32,64}.spv, with spir and spir64 respectively:

     clang --target=spir -cl-std=CL2.0 -O0 \
       -Xclang -finclude-default-header -emit-llvm -c \
       test_spirv_builtins.cl -o builtins.bc
     llvm-spirv --spirv-max-version=1.0 builtins.bc \
       -o test_spirv_builtins32.spv

   The double precision and sub-group kernels are in
   test_spirv_builtins_{fp64,subgroups}.cl, so that devices without these
   features still run the others.
*/

/* size_t parameters */
kernel void
size_t_args (global const float *f, global const half *h, global half *ho,
             global float *out, local float *l)
{
  float4 v = vload4 (1, f);
  vstore4 (v * 2.0f, 1, out);
  out[8] = vload_half (2, h);
  vstore_half (out[8] + 1.0f, 3, ho);

  prefetch (f, 4);
  event_t e = async_work_group_copy (l, f, 8, 0);
  wait_group_events (1, &e);
  out[9] = l[7];
  e = async_work_group_strided_copy (l, f, 4, 2, 0);
  wait_group_events (1, &e);
  out[10] = l[3];
}

/* 64-bit vector arguments, coerced to i64 on i386 */
kernel void
small_vectors (global char8 *c, global short4 *s, global int2 *i)
{
  c[2] = add_sat (c[0], c[1]);
  s[2] = max (s[0], s[1]);
  i[2] = hadd (i[0], i[1]);
}

/* vectors that x86-64 passes in memory, and vec3 */
kernel void
wide_vectors (global float8 *f8, global float16 *f16, global float *f3)
{
  f8[3] = fma (f8[0], f8[1], f8[2]);
  f16[2] = fmax (f16[0], f16[1]);
  float3 a = vload3 (0, f3);
  float3 b = vload3 (1, f3);
  vstore3 (cross (a, b), 2, f3);
  f3[9] = dot (a, b);
}

/* scoped atomics and barriers */
kernel void
sync (global atomic_int *counter, local int *tmp, global int *out)
{
  int lid = get_local_id (0);
  tmp[lid] = lid;
  work_group_barrier (CLK_LOCAL_MEM_FENCE, memory_scope_work_group);
  out[get_global_id (0)] = tmp[(lid + 1) % get_local_size (0)];
  atomic_fetch_add_explicit (counter, 1, memory_order_relaxed,
                             memory_scope_device);
}
