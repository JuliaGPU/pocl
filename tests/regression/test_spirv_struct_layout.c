// Copyright (c) 2026 PoCL developers
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

/*
  SPIR-V modules use the OpenCL C layout of structs, where long and double
  are 8-byte aligned, without padding the struct types explicitly. PoCL
  replaces the SPIR datalayout with the host's, which on i386 aligns long and
  double to 4 bytes only, so the kernels used to access fields, struct
  arguments and arrays of structs at the wrong offsets, while sizeof and
  offsetof (folded by the SPIR compiler) still reported the SPIR layout.

  The kernels are in test_spirv_struct_layout.cl and .ll. The host side
  writes and reads the structs as bytes at the OpenCL C offsets.
*/

#include "pocl_opencl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <unistd.h>
#endif

#define CHECK_ERROR(err)                                                       \
  if (err != CL_SUCCESS) {                                                     \
    printf("OpenCL Error %d at %s:%d\n", err, __FILE__, __LINE__);             \
    return 1;                                                                  \
  }

#define N 3
#define S_SIZE 24
#define U_SIZE 56
#define P_SIZE 32

static int Failures = 0;

static void put_i32(unsigned char *buf, size_t off, cl_int v) {
  memcpy(buf + off, &v, sizeof(v));
}
static void put_i64(unsigned char *buf, size_t off, cl_long v) {
  memcpy(buf + off, &v, sizeof(v));
}
static void put_f64(unsigned char *buf, size_t off, cl_double v) {
  memcpy(buf + off, &v, sizeof(v));
}

static void check_i64(const char *what, int idx, const unsigned char *buf,
                      size_t off, size_t size, cl_long expected) {
  cl_long v = 0;
  if (size == 1)
    v = (cl_char)buf[off];
  else if (size == 4) {
    cl_int i;
    memcpy(&i, buf + off, sizeof(i));
    v = i;
  } else
    memcpy(&v, buf + off, sizeof(v));
  if (v != expected) {
    printf("%s[%d]: got %lld, expected %lld\n", what, idx, (long long)v,
           (long long)expected);
    Failures++;
  }
}

static void check_f64(const char *what, int idx, const unsigned char *buf,
                      size_t off, cl_double expected) {
  cl_double v;
  memcpy(&v, buf + off, sizeof(v));
  if (v != expected) {
    printf("%s[%d]: got %g, expected %g\n", what, idx, v, expected);
    Failures++;
  }
}

/* T: d at 0, i at 8. */
static void put_t(unsigned char *buf, size_t off, cl_double d, cl_int i) {
  put_f64(buf, off, d);
  put_i32(buf, off + 8, i);
}
static void check_t(const char *what, int idx, const unsigned char *buf,
                    size_t off, cl_double d, cl_int i) {
  check_f64(what, idx, buf, off, d);
  check_i64(what, idx, buf, off + 8, 4, i);
}

static int run_layout(cl_context context, cl_command_queue queue,
                      cl_program program) {
  cl_int err;
  cl_kernel kernel = clCreateKernel(program, "layout", &err);
  CHECK_ERROR(err);

  /* S: c at 0, l at 8, d at 16. */
  unsigned char sv[S_SIZE] = {0};
  sv[0] = 3;
  put_i64(sv, 8, 0x100000000LL);
  put_f64(sv, 16, 0.5);

  /* U: a at 0, t[2] at 8, c at 40, l at 48. */
  unsigned char uv[U_SIZE] = {0};
  put_i32(uv, 0, 100);
  put_t(uv, 8, 1.25, 7);
  put_t(uv, 24, 2.25, 8);
  uv[40] = 9;
  put_i64(uv, 48, 1000);

  cl_mem s = clCreateBuffer(context, CL_MEM_READ_WRITE, N * S_SIZE, NULL, &err);
  CHECK_ERROR(err);
  cl_mem u = clCreateBuffer(context, CL_MEM_READ_WRITE, N * U_SIZE, NULL, &err);
  CHECK_ERROR(err);
  cl_mem out =
      clCreateBuffer(context, CL_MEM_READ_WRITE, 9 * sizeof(cl_long), NULL, &err);
  CHECK_ERROR(err);

  CHECK_ERROR(clSetKernelArg(kernel, 0, sizeof(cl_mem), &s));
  CHECK_ERROR(clSetKernelArg(kernel, 1, S_SIZE, sv));
  CHECK_ERROR(clSetKernelArg(kernel, 2, sizeof(cl_mem), &u));
  CHECK_ERROR(clSetKernelArg(kernel, 3, U_SIZE, uv));
  CHECK_ERROR(clSetKernelArg(kernel, 4, sizeof(cl_mem), &out));

  size_t global = N;
  CHECK_ERROR(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0,
                                     NULL, NULL));

  unsigned char s_res[N * S_SIZE], u_res[N * U_SIZE];
  cl_long out_res[9];
  CHECK_ERROR(clEnqueueReadBuffer(queue, s, CL_TRUE, 0, sizeof(s_res), s_res, 0,
                                  NULL, NULL));
  CHECK_ERROR(clEnqueueReadBuffer(queue, u, CL_TRUE, 0, sizeof(u_res), u_res, 0,
                                  NULL, NULL));
  CHECK_ERROR(clEnqueueReadBuffer(queue, out, CL_TRUE, 0, sizeof(out_res),
                                  out_res, 0, NULL, NULL));

  /* priv[k] = {t[k % 2].d * (k + 1), t[k % 2].i + k} */
  const cl_double priv_d[3] = {1.25, 4.5, 3.75};
  const cl_int priv_i[3] = {7, 9, 9};
  for (int i = 0; i < N; i++) {
    const unsigned char *si = s_res + i * S_SIZE;
    check_i64("s.c", i, si, 0, 1, 3 + i);
    check_i64("s.l", i, si, 8, 8, 0x100000000LL + i);
    check_f64("s.d", i, si, 16, 0.5 + i);

    const unsigned char *ui = u_res + i * U_SIZE;
    check_i64("u.a", i, ui, 0, 4, 100);
    check_t("u.t[0]", i, ui, 8, 1.25, 7);
    check_t("u.t[1]", i, ui, 24, priv_d[i % 3], priv_i[i % 3]);
    check_i64("u.c", i, ui, 40, 1, 9);
    check_i64("u.l", i, ui, 48, 8, 1007);
  }

  /* sizeof and offsetof are folded by the SPIR compiler, so these held
     before too; the last two read a constant initializer. */
  const cl_long expected_out[9] = {24, 8, 16, 16, 56, 8, 48, 5, 4};
  for (int i = 0; i < 9; i++)
    check_i64("out", i, (const unsigned char *)out_res, i * sizeof(cl_long), 8,
              expected_out[i]);

  clReleaseMemObject(s);
  clReleaseMemObject(u);
  clReleaseMemObject(out);
  clReleaseKernel(kernel);
  return 0;
}

static int run_aggregate(cl_context context, cl_command_queue queue,
                         cl_program program) {
  cl_int err;
  cl_kernel kernel = clCreateKernel(program, "aggregate", &err);
  CHECK_ERROR(err);

  /* P: a at 0, t at 8, l at 24. */
  unsigned char io[3 * P_SIZE] = {0};
  put_i32(io, 0, 1);
  put_t(io, 8, 0.25, 2);
  put_i64(io, 24, 3);

  cl_mem buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              sizeof(io), io, &err);
  CHECK_ERROR(err);
  CHECK_ERROR(clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf));

  size_t global = 1;
  CHECK_ERROR(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0,
                                     NULL, NULL));
  CHECK_ERROR(clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, sizeof(io), io, 0,
                                  NULL, NULL));

  check_i64("p.a", 0, io, 0, 4, 2);
  check_t("p.t", 0, io, 8, 0.5, 2);
  check_i64("p.l", 0, io, 24, 8, 103);
  check_i64("p.a", 1, io, P_SIZE + 0, 4, 10);
  check_t("p.t", 1, io, P_SIZE + 8, 1.5, 11);
  check_i64("p.l", 1, io, P_SIZE + 24, 8, 12);
  check_i64("p.a", 2, io, 2 * P_SIZE + 0, 4, 20);
  check_t("p.t", 2, io, 2 * P_SIZE + 8, 2.5, 21);
  check_i64("p.l", 2, io, 2 * P_SIZE + 24, 8, 22);

  clReleaseMemObject(buf);
  clReleaseKernel(kernel);
  return 0;
}

static int run_constant_gep(cl_context context, cl_command_queue queue,
                            cl_program program) {
  cl_int err;
  cl_kernel kernel = clCreateKernel(program, "constant_gep", &err);
  CHECK_ERROR(err);
  cl_int out = 0;
  cl_mem buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              sizeof(out), &out, &err);
  CHECK_ERROR(err);
  CHECK_ERROR(clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf));
  size_t global = 1;
  CHECK_ERROR(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0,
                                     NULL, NULL));
  CHECK_ERROR(clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, sizeof(out), &out, 0,
                                  NULL, NULL));
  check_i64("tarr[1].i", 0, (const unsigned char *)&out, 0, 4, 2);
  clReleaseMemObject(buf);
  clReleaseKernel(kernel);
  return 0;
}

int main(int argc, char **argv) {
  cl_uint platform_index = argc > 1 ? (cl_uint)atoi(argv[1]) : 0;
  cl_int err;

#ifndef _WIN32
  alarm(300);
#endif

  cl_uint num_platforms;
  CHECK_ERROR(clGetPlatformIDs(0, NULL, &num_platforms));
  if (platform_index >= num_platforms) {
    printf("Platform index %u out of range\n", platform_index);
    return 1;
  }
  cl_platform_id *platforms = malloc(sizeof(cl_platform_id) * num_platforms);
  CHECK_ERROR(clGetPlatformIDs(num_platforms, platforms, NULL));
  cl_platform_id platform = platforms[platform_index];
  free(platforms);

  cl_device_id device;
  CHECK_ERROR(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL));

  if (!poclu_device_supports_il(device, "SPIR-V_1.0")) {
    printf("SKIP: The test requires support for SPIR-V 1.0\n");
    return 77;
  }
  cl_device_fp_config fp64_config = 0;
  CHECK_ERROR(clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG,
                              sizeof(fp64_config), &fp64_config, NULL));
  if (fp64_config == 0) {
    printf("SKIP: The test requires double precision support\n");
    return 77;
  }

  const char *path = poclu_device_address_bits(device) == 32
                         ? SRCDIR "/test_spirv_struct_layout32.spv"
                         : SRCDIR "/test_spirv_struct_layout64.spv";
  size_t size;
  char *binary = poclu_read_binfile(path, &size);
  if (binary == NULL) {
    printf("Failed to open %s\n", path);
    return 1;
  }

  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CHECK_ERROR(err);
  cl_command_queue queue =
      clCreateCommandQueueWithProperties(context, device, NULL, &err);
  CHECK_ERROR(err);

  cl_program program = clCreateProgramWithIL(context, binary, size, &err);
  CHECK_ERROR(err);
  free(binary);
  err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    poclu_show_program_build_log(program);
    CHECK_ERROR(err);
  }

  if (run_layout(context, queue, program) ||
      run_aggregate(context, queue, program) ||
      run_constant_gep(context, queue, program))
    return 1;

  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  if (Failures) {
    printf("FAIL: %d mismatches\n", Failures);
    return 1;
  }
  printf("OK\n");
  return 0;
}
