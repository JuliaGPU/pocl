/* Tests for the cl_pocl_host_kernel extension.

   Copyright (c) 2026 PoCL developers

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#define _GNU_SOURCE /* feenableexcept */
#include "pocl_opencl.h"

#include <fenv.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "poclu.h"

static const char *source
  = "global int base = 1000;\n"
    "\n"
    "kernel void ids (global int *out, int add) {\n"
    "  size_t x = get_global_id (0), y = get_global_id (1);\n"
    "  out[y * get_global_size (0) + x - get_global_offset (0)\n"
    "      - get_global_offset (1) * get_global_size (0)]\n"
    "      = base + add + (int)(y * 100 + x);\n"
    "}\n"
    "\n"
    "kernel void group_sums (global const int *in, global int *out,\n"
    "                        local int *scratch) {\n"
    "  local int total;\n"
    "  size_t l = get_local_id (0);\n"
    "  scratch[l] = in[get_global_id (0)];\n"
    "  if (l == 0)\n"
    "    total = 0;\n"
    "  barrier (CLK_LOCAL_MEM_FENCE);\n"
    "  if (l == 0)\n"
    "    for (size_t i = 0; i < get_local_size (0); ++i)\n"
    "      total += scratch[i];\n"
    "  barrier (CLK_LOCAL_MEM_FENCE);\n"
    "  if (l == get_local_size (0) - 1)\n"
    "    out[get_group_id (0)] = total;\n"
    "}\n"
    "\n"
    "kernel void rounding (global float *out, float a, float b) {\n"
    "  out[0] = a / b;\n"
    "}\n"
    "\n"
    "kernel void trap (global int *out) {\n"
    "  size_t i = get_global_id (0);\n"
    "  if (i == 3)\n"
    "    __builtin_trap ();\n"
    "  out[i] = 1;\n"
    "}\n"
    "\n"
    "kernel void vector (global float16 *out, float16 v) {\n"
    "  out[0] = v + v;\n"
    "}\n"
    "\n"
    "kernel void group_ids (global int *out) {\n"
    "  size_t x = get_group_id (0), y = get_group_id (1), z = get_group_id (2);\n"
    "  if (x < get_num_groups (0) && y < get_num_groups (1)\n"
    "      && z < get_num_groups (2))\n"
    "    out[(z * get_num_groups (1) + y) * get_num_groups (0) + x] = 1;\n"
    "}\n"
    "\n"
    "kernel void big_private (global int *out, int x) {\n"
    "  volatile int a[256];\n"
    "  for (int i = 0; i < 256; ++i)\n"
    "    a[i] = i * x;\n"
    "  out[0] = a[x & 255];\n"
    "}\n"
    "\n"
    "#ifndef NO_IMAGES\n"
    "kernel void image (read_only image2d_t img) {}\n"
    "#endif\n";

static clCreateHostKernelPOCL_fn create_hk;
static clGetHostKernelInfoPOCL_fn get_hk_info;
static clRetainHostKernelPOCL_fn retain_hk;
static clReleaseHostKernelPOCL_fn release_hk;
static clRunHostKernelPOCL_fn run_hk;

static void *
alloc_scratch (cl_host_kernel_pocl hk)
{
  size_t size, align;
  cl_int err = get_hk_info (hk, CL_HOST_KERNEL_SCRATCH_SIZE_POCL,
                            sizeof (size), &size, NULL);
  if (err != CL_SUCCESS)
    return NULL;
  err = get_hk_info (hk, CL_HOST_KERNEL_SCRATCH_ALIGNMENT_POCL, sizeof (align),
                     &align, NULL);
  if (err != CL_SUCCESS)
    return NULL;
  void *p = NULL;
  if (posix_memalign (&p, align, size) != 0)
    return NULL;
  return p;
}

/* A 2D grid split into ranges of work-groups that several threads run
   concurrently through one handle, each with its own scratch area. */

#define NX 64
#define NY 24
#define LX 8
#define LY 2
#define NTHREADS 4

struct ids_job
{
  cl_host_kernel_pocl hk;
  const void *const *args;
  const size_t *num_groups;
  const size_t *offset;
  size_t first, count;
  cl_int err;
};

static void *
run_ids_job (void *p)
{
  struct ids_job *job = (struct ids_job *)p;
  void *scratch = alloc_scratch (job->hk);
  /* Each thread runs its range in pieces of 5 groups. */
  for (size_t i = 0; i < job->count && job->err == CL_SUCCESS; i += 5)
    {
      size_t n = job->count - i < 5 ? job->count - i : 5;
      job->err = run_hk (job->hk, job->args, job->num_groups, job->offset,
                         job->first + i, n, scratch);
    }
  free (scratch);
  return NULL;
}

static int
test_ids (cl_kernel kernel, cl_device_id device, const size_t *offset)
{
  const size_t local[2] = { LX, LY };
  cl_int err;
  cl_host_kernel_pocl hk = create_hk (kernel, device, 2, local, &err);
  CHECK_CL_ERROR (err);

  int *out = malloc (sizeof (int) * NX * NY);
  for (size_t i = 0; i < NX * NY; ++i)
    out[i] = -1;
  int add = 7;
  const void *args[2] = { &out, &add };
  const size_t num_groups[2] = { NX / LX, NY / LY };
  const size_t total = num_groups[0] * num_groups[1];

  pthread_t threads[NTHREADS];
  struct ids_job jobs[NTHREADS];
  for (int t = 0; t < NTHREADS; ++t)
    {
      size_t first = total * t / NTHREADS;
      size_t last = total * (t + 1) / NTHREADS;
      struct ids_job job
        = { hk, args, num_groups, offset, first, last - first, CL_SUCCESS };
      jobs[t] = job;
      TEST_ASSERT (pthread_create (&threads[t], NULL, run_ids_job, &jobs[t])
                   == 0);
    }
  for (int t = 0; t < NTHREADS; ++t)
    {
      pthread_join (threads[t], NULL);
      CHECK_CL_ERROR (jobs[t].err);
    }

  size_t ox = offset ? offset[0] : 0, oy = offset ? offset[1] : 0;
  for (size_t y = 0; y < NY; ++y)
    for (size_t x = 0; x < NX; ++x)
      {
        int expected = 1000 + add + (int)((y + oy) * 100 + (x + ox));
        if (out[y * NX + x] != expected)
          {
            printf ("FAIL: ids[%zu, %zu] = %d, expected %d\n", x, y,
                    out[y * NX + x], expected);
            return EXIT_FAILURE;
          }
      }

  free (out);
  CHECK_CL_ERROR (release_hk (hk));
  return EXIT_SUCCESS;
}

/* __local arguments and automatic locals, with barriers. The local argument
   size is taken at creation, so changing it afterwards has no effect. */
static int
test_group_sums (cl_kernel kernel, cl_device_id device)
{
  enum
  {
    L = 32,
    G = 16
  };
  cl_int err;
  CHECK_CL_ERROR (clSetKernelArg (kernel, 2, L * sizeof (cl_int), NULL));
  const size_t local = L;
  cl_host_kernel_pocl hk = create_hk (kernel, device, 1, &local, &err);
  CHECK_CL_ERROR (err);
  CHECK_CL_ERROR (clSetKernelArg (kernel, 2, sizeof (cl_int), NULL));

  int *in = malloc (sizeof (int) * L * G);
  int *out = malloc (sizeof (int) * G);
  for (int i = 0; i < L * G; ++i)
    in[i] = i;
  const void *args[3] = { &in, &out, NULL };
  const size_t num_groups = G;
  void *scratch = alloc_scratch (hk);
  CHECK_CL_ERROR (run_hk (hk, args, &num_groups, NULL, 0, G, scratch));

  for (int g = 0; g < G; ++g)
    {
      int expected = 0;
      for (int i = 0; i < L; ++i)
        expected += g * L + i;
      if (out[g] != expected)
        {
          printf ("FAIL: group_sums[%d] = %d, expected %d\n", g, out[g],
                  expected);
          return EXIT_FAILURE;
        }
    }

  free (scratch);
  free (in);
  free (out);
  CHECK_CL_ERROR (release_hk (hk));
  return EXIT_SUCCESS;
}

/* The kernel runs with round-to-nearest, and the caller's rounding mode and
   exception flags survive the call. */
static int
test_fp_env (cl_kernel kernel, cl_device_id device)
{
#if defined(__x86_64__) || defined(__i386__)
  cl_int err;
  const size_t local = 1;
  cl_host_kernel_pocl hk = create_hk (kernel, device, 1, &local, &err);
  CHECK_CL_ERROR (err);

  float out = 0.0f, a = 1.0f, b = 3.0f;
  float *outp = &out;
  const void *args[3] = { &outp, &a, &b };
  const size_t num_groups = 1;
  void *scratch = alloc_scratch (hk);

  TEST_ASSERT (fesetround (FE_UPWARD) == 0);
  feclearexcept (FE_ALL_EXCEPT);
  feraiseexcept (FE_DIVBYZERO);
  err = run_hk (hk, args, &num_groups, NULL, 0, 1, scratch);
  int rounding = fegetround ();
  int divbyzero = fetestexcept (FE_DIVBYZERO);
  int inexact = fetestexcept (FE_INEXACT);
  fesetround (FE_TONEAREST);
  feclearexcept (FE_ALL_EXCEPT);
  CHECK_CL_ERROR (err);

  volatile float va = 1.0f, vb = 3.0f;
  float nearest = va / vb;
  TEST_ASSERT (out == nearest);
  TEST_ASSERT (rounding == FE_UPWARD);
  TEST_ASSERT (divbyzero);
  TEST_ASSERT (!inexact);

#ifdef __GLIBC__
  /* Exceptions the caller unmasked don't trap in the kernel. */
  b = 0.0f;
  feenableexcept (FE_DIVBYZERO);
  err = run_hk (hk, args, &num_groups, NULL, 0, 1, scratch);
  int enabled = fegetexcept ();
  fedisableexcept (FE_ALL_EXCEPT);
  feclearexcept (FE_ALL_EXCEPT);
  CHECK_CL_ERROR (err);
  TEST_ASSERT (out == INFINITY);
  TEST_ASSERT (enabled == FE_DIVBYZERO);
#endif

  free (scratch);
  CHECK_CL_ERROR (release_hk (hk));
#endif
  return EXIT_SUCCESS;
}

/* A trapping work-item doesn't end the call, and is reported exactly when the
   same launch through a command queue fails. */
static int
test_trap (cl_kernel kernel,
           cl_context context,
           cl_command_queue queue,
           cl_device_id device)
{
  cl_int err;
  const size_t local = 4, global = 16;

  cl_mem buf = clCreateBuffer (context, CL_MEM_READ_WRITE, 16 * sizeof (int),
                               NULL, &err);
  CHECK_CL_ERROR (err);
  CHECK_CL_ERROR (clSetKernelArg (kernel, 0, sizeof (cl_mem), &buf));
  cl_event ev;
  CHECK_CL_ERROR (clEnqueueNDRangeKernel (queue, kernel, 1, NULL, &global,
                                          &local, 0, NULL, &ev));
  clWaitForEvents (1, &ev);
  cl_int status;
  CHECK_CL_ERROR (clGetEventInfo (ev, CL_EVENT_COMMAND_EXECUTION_STATUS,
                                  sizeof (status), &status, NULL));
  CHECK_CL_ERROR (clReleaseEvent (ev));
  CHECK_CL_ERROR (clReleaseMemObject (buf));
  clFinish (queue);

  cl_host_kernel_pocl hk = create_hk (kernel, device, 1, &local, &err);
  CHECK_CL_ERROR (err);
  int out[16] = { 0 };
  int *outp = out;
  const void *args[1] = { &outp };
  const size_t num_groups = global / local;
  void *scratch = alloc_scratch (hk);
  err = run_hk (hk, args, &num_groups, NULL, 0, num_groups, scratch);
  TEST_ASSERT (err == (status < 0 ? CL_OUT_OF_RESOURCES : CL_SUCCESS));
  for (int i = 4; i < 16; ++i)
    TEST_ASSERT (out[i] == 1);

  free (scratch);
  CHECK_CL_ERROR (release_hk (hk));
  return EXIT_SUCCESS;
}

/* A float16 is loaded with 64-byte alignment, so a value that isn't aligned
   must be copied first. */
static int
test_arg_alignment (cl_kernel kernel, cl_device_id device)
{
  const size_t local = 1, num_groups = 1;
  cl_int err;
  cl_host_kernel_pocl hk = create_hk (kernel, device, 1, &local, &err);
  CHECK_CL_ERROR (err);
  void *scratch = alloc_scratch (hk);

  cl_float16 out;
  cl_float16 *outp = &out;
  _Alignas (64) unsigned char storage[64 + sizeof (cl_float16)];
  for (int offset = 0; offset <= 4; offset += 4)
    {
      cl_float16 v = { { 0 } };
      v.s[0] = 1;
      v.s[15] = 2;
      memcpy (storage + offset, &v, sizeof (v));
      const void *args[2] = { &outp, storage + offset };
      memset (&out, 0, sizeof (out));
      CHECK_CL_ERROR (run_hk (hk, args, &num_groups, NULL, 0, 1, scratch));
      TEST_ASSERT (out.s[0] == 2 && out.s[15] == 4);
    }

  free (scratch);
  CHECK_CL_ERROR (release_hk (hk));
  return EXIT_SUCCESS;
}

/* On Win64, group_z is passed on the stack. Dirty it first, so a group id that
   is only partially written shows. */
static __attribute__ ((noinline)) void
dirty_stack (void)
{
  volatile unsigned char junk[16384];
  for (size_t i = 0; i < sizeof (junk); ++i)
    junk[i] = 0xff;
}

static int
test_group_ids (cl_kernel kernel, cl_device_id device)
{
  const size_t local[3] = { 1, 1, 1 }, num_groups[3] = { 2, 3, 4 };
  enum
  {
    N = 2 * 3 * 4
  };
  cl_int err;
  cl_host_kernel_pocl hk = create_hk (kernel, device, 3, local, &err);
  CHECK_CL_ERROR (err);
  void *scratch = alloc_scratch (hk);

  int out[N] = { 0 };
  int *outp = out;
  const void *args[1] = { &outp };
  dirty_stack ();
  CHECK_CL_ERROR (run_hk (hk, args, num_groups, NULL, 0, N, scratch));
  for (int i = 0; i < N; ++i)
    if (out[i] != 1)
      {
        printf ("FAIL: group %d didn't run with its ids\n", i);
        return EXIT_FAILURE;
      }

  free (scratch);
  CHECK_CL_ERROR (release_hk (hk));
  return EXIT_SUCCESS;
}

int
main (void)
{
  cl_int err;
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;

  CHECK_CL_ERROR (
    poclu_get_any_device2 (&context, &device, &queue, &platform));

  if (!poclu_supports_extension (device, "cl_pocl_host_kernel"))
    {
      printf ("SKIP: the device doesn't support cl_pocl_host_kernel\n");
      return 77;
    }

  create_hk
    = (clCreateHostKernelPOCL_fn)clGetExtensionFunctionAddressForPlatform (
      platform, "clCreateHostKernelPOCL");
  get_hk_info
    = (clGetHostKernelInfoPOCL_fn)clGetExtensionFunctionAddressForPlatform (
      platform, "clGetHostKernelInfoPOCL");
  retain_hk
    = (clRetainHostKernelPOCL_fn)clGetExtensionFunctionAddressForPlatform (
      platform, "clRetainHostKernelPOCL");
  release_hk
    = (clReleaseHostKernelPOCL_fn)clGetExtensionFunctionAddressForPlatform (
      platform, "clReleaseHostKernelPOCL");
  run_hk = (clRunHostKernelPOCL_fn)clGetExtensionFunctionAddressForPlatform (
    platform, "clRunHostKernelPOCL");
  TEST_ASSERT (create_hk && get_hk_info && retain_hk && release_hk && run_hk);

  cl_bool images = CL_FALSE;
  CHECK_CL_ERROR (clGetDeviceInfo (device, CL_DEVICE_IMAGE_SUPPORT,
                                   sizeof (images), &images, NULL));
  const char *options = images ? "-cl-std=CL3.0" : "-cl-std=CL3.0 -DNO_IMAGES";
  cl_program program
    = clCreateProgramWithSource (context, 1, &source, NULL, &err);
  CHECK_CL_ERROR (err);
  CHECK_CL_ERROR (clBuildProgram (program, 1, &device, options, NULL, NULL));

  cl_kernel ids = clCreateKernel (program, "ids", &err);
  CHECK_CL_ERROR (err);
  cl_kernel group_sums = clCreateKernel (program, "group_sums", &err);
  CHECK_CL_ERROR (err);
  cl_kernel rounding = clCreateKernel (program, "rounding", &err);
  CHECK_CL_ERROR (err);
  cl_kernel trap = clCreateKernel (program, "trap", &err);
  CHECK_CL_ERROR (err);
  cl_kernel vector = clCreateKernel (program, "vector", &err);
  CHECK_CL_ERROR (err);
  cl_kernel group_ids = clCreateKernel (program, "group_ids", &err);
  CHECK_CL_ERROR (err);

  if (test_ids (ids, device, NULL) || test_group_sums (group_sums, device)
      || test_fp_env (rounding, device)
      || test_trap (trap, context, queue, device)
      || test_arg_alignment (vector, device)
      || test_group_ids (group_ids, device))
    return EXIT_FAILURE;
  const size_t offset[2] = { 3, 70000 };
  if (test_ids (ids, device, offset))
    return EXIT_FAILURE;

  /* The stack estimate, which decides whether a work-group fits on the
     calling thread's stack, is reported per work-item. */
  cl_kernel big_private = clCreateKernel (program, "big_private", &err);
  CHECK_CL_ERROR (err);
  cl_ulong private_mem = 0;
  CHECK_CL_ERROR (clGetKernelWorkGroupInfo (
      big_private, device, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof (private_mem),
      &private_mem, NULL));
  TEST_ASSERT (private_mem >= 256 * sizeof (cl_int));
  CHECK_CL_ERROR (clReleaseKernel (big_private));

  /* Invalid uses. */
  size_t bad_local[4] = { 1, 1, 1, 1 };
  TEST_ASSERT (create_hk (ids, device, 4, bad_local, &err) == NULL);
  TEST_ASSERT (err == CL_INVALID_WORK_DIMENSION);
  size_t huge_local = (size_t)1 << 40;
  TEST_ASSERT (create_hk (ids, device, 1, &huge_local, &err) == NULL);
  TEST_ASSERT (err == CL_INVALID_WORK_ITEM_SIZE);
  cl_kernel unset = clCreateKernel (program, "group_sums", &err);
  CHECK_CL_ERROR (err);
  const size_t local = 4;
  TEST_ASSERT (create_hk (unset, device, 1, &local, &err) == NULL);
  TEST_ASSERT (err == CL_INVALID_KERNEL_ARGS);
  CHECK_CL_ERROR (clReleaseKernel (unset));
  if (images)
    {
      cl_kernel image = clCreateKernel (program, "image", &err);
      CHECK_CL_ERROR (err);
      TEST_ASSERT (create_hk (image, device, 1, &local, &err) == NULL);
      TEST_ASSERT (err == CL_INVALID_KERNEL);
      CHECK_CL_ERROR (clReleaseKernel (image));
    }

  /* The handle outlives the kernel and the program, and keeps the program's
     global variables. */
  const size_t ids_local[2] = { LX, LY };
  cl_host_kernel_pocl hk = create_hk (ids, device, 2, ids_local, &err);
  CHECK_CL_ERROR (err);
  CHECK_CL_ERROR (retain_hk (hk));
  cl_uint refcount = 0;
  CHECK_CL_ERROR (get_hk_info (hk, CL_HOST_KERNEL_REFERENCE_COUNT_POCL,
                               sizeof (refcount), &refcount, NULL));
  TEST_ASSERT (refcount == 2);
  CHECK_CL_ERROR (release_hk (hk));

  /* A __local argument larger than the device's local memory. */
  cl_kernel big_local = clCreateKernel (program, "group_sums", &err);
  CHECK_CL_ERROR (err);
  if (clSetKernelArg (big_local, 2, SIZE_MAX, NULL) == CL_SUCCESS)
    {
      const size_t one = 1;
      TEST_ASSERT (create_hk (big_local, device, 1, &one, &err) == NULL);
      TEST_ASSERT (err == CL_OUT_OF_RESOURCES);
    }
  CHECK_CL_ERROR (clReleaseKernel (big_local));

  CHECK_CL_ERROR (clReleaseKernel (ids));
  CHECK_CL_ERROR (clReleaseKernel (group_sums));
  CHECK_CL_ERROR (clReleaseKernel (rounding));
  CHECK_CL_ERROR (clReleaseKernel (trap));
  CHECK_CL_ERROR (clReleaseKernel (vector));
  CHECK_CL_ERROR (clReleaseKernel (group_ids));
  /* While handles exist, the program keeps a kernel and can't be rebuilt. */
  TEST_ASSERT (clBuildProgram (program, 1, &device, options, NULL, NULL)
               == CL_INVALID_OPERATION);

  CHECK_CL_ERROR (clReleaseProgram (program));

  int out[LX * LY];
  int *outp = out;
  int add = 1;
  const void *args[2] = { &outp, &add };
  const size_t num_groups[2] = { 1, 1 };
  void *scratch = alloc_scratch (hk);
  TEST_ASSERT (run_hk (hk, args, num_groups, NULL, 1, 1, scratch)
               == CL_INVALID_VALUE);
  CHECK_CL_ERROR (run_hk (hk, args, num_groups, NULL, 0, 1, scratch));
  TEST_ASSERT (out[LX + 1] == 1000 + add + 101);
  free (scratch);
  CHECK_CL_ERROR (release_hk (hk));

  CHECK_CL_ERROR (clReleaseCommandQueue (queue));
  CHECK_CL_ERROR (clReleaseContext (context));
  CHECK_CL_ERROR (clUnloadPlatformCompiler (platform));

  printf ("OK\n");
  return EXIT_SUCCESS;
}
