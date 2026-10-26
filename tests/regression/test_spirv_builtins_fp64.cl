/* Double precision kernels for test_spirv_builtins.c, built like
   test_spirv_builtins.cl to test_spirv_builtins_fp64{32,64}.spv. */

kernel void
fp64 (global double4 *d4, global double *d3)
{
  d4[3] = fma (d4[0], d4[1], d4[2]);
  double3 a = vload3 (0, d3);
  double3 b = vload3 (1, d3);
  vstore3 (cross (a, b), 2, d3);
}
