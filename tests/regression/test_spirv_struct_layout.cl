/* Kernels for test_spirv_struct_layout.c. The struct types aren't padded
   explicitly in SPIR, where long and double are 8-byte aligned. Built to
   test_spirv_struct_layout{32,64}.spv together with
   test_spirv_struct_layout.ll, for N in 32 and 64 (with spir64 for 64):

     clang --target=spir -cl-std=CL2.0 -O0 -emit-llvm -c \
       test_spirv_struct_layout.cl -o cl.bc
     llvm-as test_spirv_struct_layout.ll -o ll.bc
     llvm-link --suppress-warnings cl.bc ll.bc -o linked.bc
     llvm-spirv --spirv-max-version=1.0 linked.bc \
       -o test_spirv_struct_layoutN.spv
*/

/* interior padding: c at 0, l at 8, d at 16, size 24 */
typedef struct
{
  char c;
  long l;
  double d;
} S;

/* tail padding: d at 0, i at 8, size 16 */
typedef struct
{
  double d;
  int i;
} T;

/* nested: a at 0, t at 8, c at 40, l at 48, size 56 */
typedef struct
{
  int a;
  T t[2];
  char c;
  long l;
} U;

constant U cu = { 1, { { 2.5, 3 }, { 4.5, 5 } }, 6, 7 };
constant long *cu_l = &cu.l;

kernel void
layout (global S *s, S sv, global U *u, U uv, global long *out)
{
  int i = get_global_id (0);
  s[i].c = sv.c + i;
  s[i].l = sv.l + i;
  s[i].d = sv.d + i;

  T priv[3];
  for (int k = 0; k < 3; k++)
    {
      priv[k].d = uv.t[k % 2].d * (k + 1);
      priv[k].i = uv.t[k % 2].i + k;
    }
  u[i] = uv;
  u[i].t[1] = priv[i % 3];
  u[i].l += *cu_l;

  if (i == 0)
    {
      out[0] = sizeof (S);
      out[1] = __builtin_offsetof (S, l);
      out[2] = __builtin_offsetof (S, d);
      out[3] = sizeof (T);
      out[4] = sizeof (U);
      out[5] = __builtin_offsetof (U, t);
      out[6] = __builtin_offsetof (U, l);
      out[7] = cu.t[1].i;
      out[8] = (long)cu.t[1].d;
    }
}
