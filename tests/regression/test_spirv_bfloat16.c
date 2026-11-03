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
  A kernel using SPV_KHR_bfloat16 to store, load and convert bfloat16 values,
  including a bfloat16 kernel argument. The SPIR-V reader rejects modules using
  extensions that are not enabled through --spirv-ext, so before the extension
  was enabled, building the program failed. Float to bfloat16 conversions
  round to nearest even. Test source (test_spirv_bfloat16.spv, assembled with
  spirv-as --target-env spv1.0):

               OpCapability Addresses
               OpCapability Kernel
               OpCapability Int64
               OpCapability BFloat16TypeKHR
               OpExtension "SPV_KHR_bfloat16"
               OpMemoryModel Physical64 OpenCL
               OpEntryPoint Kernel %kernel "convert" %gid
               OpDecorate %gid BuiltIn GlobalInvocationId
               OpDecorate %gid Constant
      %ulong = OpTypeInt 64 0
    %v3ulong = OpTypeVector %ulong 3
  %ptr_input = OpTypePointer Input %v3ulong
       %void = OpTypeVoid
      %float = OpTypeFloat 32
     %bf16 = OpTypeFloat 16 BFloat16KHR
  %ptr_f32 = OpTypePointer CrossWorkgroup %float
 %ptr_bf16 = OpTypePointer CrossWorkgroup %bf16
         %fn = OpTypeFunction %void %ptr_bf16 %ptr_f32 %ptr_f32 %ptr_bf16 %bf16
        %gid = OpVariable %ptr_input Input
     %kernel = OpFunction %void None %fn
     %out_bf = OpFunctionParameter %ptr_bf16
      %out_f = OpFunctionParameter %ptr_f32
       %in_f = OpFunctionParameter %ptr_f32
      %in_bf = OpFunctionParameter %ptr_bf16
      %scale = OpFunctionParameter %bf16
      %entry = OpLabel
       %gid3 = OpLoad %v3ulong %gid Aligned 32
          %i = OpCompositeExtract %ulong %gid3 0
   %in_f_ptr = OpInBoundsPtrAccessChain %ptr_f32 %in_f %i
  %in_bf_ptr = OpInBoundsPtrAccessChain %ptr_bf16 %in_bf %i
 %out_bf_ptr = OpInBoundsPtrAccessChain %ptr_bf16 %out_bf %i
  %out_f_ptr = OpInBoundsPtrAccessChain %ptr_f32 %out_f %i
          %f = OpLoad %float %in_f_ptr Aligned 4
     %narrow = OpFConvert %bf16 %f
               OpStore %out_bf_ptr %narrow Aligned 2
          %b = OpLoad %bf16 %in_bf_ptr Aligned 2
       %wide = OpFConvert %float %b
    %scale_f = OpFConvert %float %scale
     %scaled = OpFMul %float %wide %scale_f
               OpStore %out_f_ptr %scaled Aligned 4
               OpReturn
               OpFunctionEnd
*/

#include "pocl_opencl.h"

#include <stdint.h>
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

#define N 16

static float bits_to_float(uint32_t u) {
  float f;
  memcpy(&f, &u, sizeof f);
  return f;
}

int main(int argc, char **argv) {
  cl_uint platform_index = argc > 1 ? (cl_uint)atoi(argv[1]) : 0;
  cl_int err;

#ifndef _WIN32
  // Fail quickly instead of tripping the much larger ctest timeout in case
  // a regressed build hangs.
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

  size_t extensions_size;
  CHECK_ERROR(clGetDeviceInfo(device, CL_DEVICE_SPIRV_EXTENSIONS_KHR, 0, NULL,
                              &extensions_size));
  const char **extensions = malloc(extensions_size);
  CHECK_ERROR(clGetDeviceInfo(device, CL_DEVICE_SPIRV_EXTENSIONS_KHR,
                              extensions_size, extensions, NULL));
  int supported = 0;
  for (size_t i = 0; i < extensions_size / sizeof(const char *); i++)
    if (strcmp(extensions[i], "SPV_KHR_bfloat16") == 0)
      supported = 1;
  free(extensions);
  if (!supported) {
#ifdef EXPECT_CPU_SPV_KHR_BFLOAT16
    // The CPU devices accept it whenever the translator does.
    cl_device_type type;
    CHECK_ERROR(
        clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, NULL));
    if (type & CL_DEVICE_TYPE_CPU) {
      printf("The CPU device does not report SPV_KHR_bfloat16\n");
      return 1;
    }
#endif
    printf("SKIP: The test requires SPV_KHR_bfloat16\n");
    return 77;
  }

  size_t capabilities_size;
  CHECK_ERROR(clGetDeviceInfo(device, CL_DEVICE_SPIRV_CAPABILITIES_KHR, 0, NULL,
                              &capabilities_size));
  cl_uint *capabilities = malloc(capabilities_size);
  CHECK_ERROR(clGetDeviceInfo(device, CL_DEVICE_SPIRV_CAPABILITIES_KHR,
                              capabilities_size, capabilities, NULL));
  int has_capability = 0;
  for (size_t i = 0; i < capabilities_size / sizeof(cl_uint); i++)
    if (capabilities[i] == 5116) // BFloat16TypeKHR
      has_capability = 1;
  free(capabilities);
  if (!has_capability) {
    printf("SPV_KHR_bfloat16 is reported without BFloat16TypeKHR\n");
    return 1;
  }

  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CHECK_ERROR(err);
  cl_queue_properties props[] = {0};
  cl_command_queue queue =
      clCreateCommandQueueWithProperties(context, device, props, &err);
  CHECK_ERROR(err);

  FILE *f = fopen(SRCDIR "/test_spirv_bfloat16.spv", "rb");
  if (!f) {
    printf("Failed to open test_spirv_bfloat16.spv\n");
    return 1;
  }
  fseek(f, 0, SEEK_END);
  size_t size = ftell(f);
  fseek(f, 0, SEEK_SET);
  unsigned char *binary = malloc(size);
  if (fread(binary, 1, size, f) != size) {
    printf("Failed to read SPIR-V\n");
    free(binary);
    fclose(f);
    return 1;
  }
  fclose(f);

  cl_program program = clCreateProgramWithIL(context, binary, size, &err);
  CHECK_ERROR(err);
  CHECK_ERROR(clBuildProgram(program, 1, &device, NULL, NULL, NULL));

  cl_kernel kernel = clCreateKernel(program, "convert", &err);
  CHECK_ERROR(err);

  // Float inputs and their bfloat16 bits, rounded to nearest even.
  static const uint32_t in_f_bits[N] = {
      0x3f800000, // 1.0
      0x3f808000, // halfway between 1.0 and its successor: to even, down
      0x3f818000, // halfway, odd below: to even, up
      0x3f808001, // just above halfway: up
      0x3f807fff, // just below halfway: down
      0xc0200000, // -2.5
      0xbf808000, // -(halfway): to even, down in magnitude
      0x7f7fffff, // FLT_MAX: overflows to infinity
      0x7f800000, // infinity
      0xff800000, // -infinity
      0x00000000, // 0.0
      0x80000000, // -0.0
      0x40490fdb, // pi
      0x3dcccccd, // 0.1
      0x477fe000, // 65504
      0x3f7fffff, // just below 1.0: up to 1.0
  };
  static const uint16_t expected_bf_bits[N] = {
      0x3f80, 0x3f80, 0x3f82, 0x3f81, 0x3f80, 0xc020, 0xbf80, 0x7f80,
      0x7f80, 0xff80, 0x0000, 0x8000, 0x4049, 0x3dcd, 0x4780, 0x3f80,
  };

  float in_f[N];
  uint16_t in_bf[N];
  for (int i = 0; i < N; i++) {
    in_f[i] = bits_to_float(in_f_bits[i]);
    // Widening is exact: the bfloat16 bits are the float's upper half.
    in_bf[i] = (uint16_t)(in_f_bits[i] >> 16);
  }
  uint16_t scale = 0xc000; // -2.0

  uint16_t out_bf[N];
  float out_f[N];
  memset(out_bf, 0xff, sizeof(out_bf));
  memset(out_f, 0xff, sizeof(out_f));

  cl_mem out_bf_buf =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(out_bf), NULL, &err);
  CHECK_ERROR(err);
  cl_mem out_f_buf =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(out_f), NULL, &err);
  CHECK_ERROR(err);
  cl_mem in_f_buf =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(in_f), in_f, &err);
  CHECK_ERROR(err);
  cl_mem in_bf_buf =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(in_bf), in_bf, &err);
  CHECK_ERROR(err);
  CHECK_ERROR(clSetKernelArg(kernel, 0, sizeof(cl_mem), &out_bf_buf));
  CHECK_ERROR(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_f_buf));
  CHECK_ERROR(clSetKernelArg(kernel, 2, sizeof(cl_mem), &in_f_buf));
  CHECK_ERROR(clSetKernelArg(kernel, 3, sizeof(cl_mem), &in_bf_buf));
  CHECK_ERROR(clSetKernelArg(kernel, 4, sizeof(scale), &scale));

  size_t global_size = N, local_size = 8;
  CHECK_ERROR(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size,
                                     &local_size, 0, NULL, NULL));
  CHECK_ERROR(clEnqueueReadBuffer(queue, out_bf_buf, CL_TRUE, 0, sizeof(out_bf),
                                  out_bf, 0, NULL, NULL));
  CHECK_ERROR(clEnqueueReadBuffer(queue, out_f_buf, CL_TRUE, 0, sizeof(out_f),
                                  out_f, 0, NULL, NULL));

  int failed = 0;
  for (int i = 0; i < N; i++) {
    if (out_bf[i] != expected_bf_bits[i]) {
      printf("bfloat16(%a) = 0x%04x, expected 0x%04x\n", in_f[i], out_bf[i],
             expected_bf_bits[i]);
      failed = 1;
    }
    float expected = bits_to_float((uint32_t)in_bf[i] << 16) * -2.0f;
    if (memcmp(&out_f[i], &expected, sizeof(float)) != 0) {
      printf("float(bfloat16 0x%04x) * -2 = %a, expected %a\n", in_bf[i],
             out_f[i], expected);
      failed = 1;
    }
  }

  clReleaseMemObject(out_bf_buf);
  clReleaseMemObject(out_f_buf);
  clReleaseMemObject(in_f_buf);
  clReleaseMemObject(in_bf_buf);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  free(binary);

  if (failed)
    return 1;
  printf("OK\n");
  return 0;
}
