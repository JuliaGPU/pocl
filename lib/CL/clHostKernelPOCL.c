/* OpenCL runtime library: cl_pocl_host_kernel

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

#include "pocl_cl.h"
#include "pocl_util.h"

/* Reserves SIZE bytes of __local memory at *OFFSET, keeping the total within
   LIMIT. */
static int
reserve_local (size_t *offset, size_t size, size_t limit)
{
  if (size > limit || *offset > limit - size)
    return 0;
  *offset += size;
  *offset = (*offset + MAX_EXTENDED_ALIGNMENT - 1)
            & ~(size_t)(MAX_EXTENDED_ALIGNMENT - 1);
  return 1;
}

static int
mul_overflows (size_t a, size_t b, size_t *res)
{
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_mul_overflow (a, b, res);
#else
  *res = a * b;
  return a != 0 && *res / a != b;
#endif
}

static int
add_overflows (size_t a, size_t b, size_t *res)
{
  *res = a + b;
  return *res < a;
}

static void
free_host_kernel (cl_host_kernel_pocl hk)
{
  if (hk->device_data != NULL)
    hk->device->ops->free_host_kernel (hk->device, hk);
  if (hk->kernel != NULL)
    POname (clReleaseKernel) (hk->kernel);
  POCL_DESTROY_OBJECT (hk);
  POCL_MEM_FREE (hk->local_offsets);
  POCL_MEM_FREE (hk->arg_alignment_masks);
  POCL_MEM_FREE (hk);
}

CL_API_ENTRY cl_host_kernel_pocl CL_API_CALL
POname (clCreateHostKernelPOCL) (cl_kernel kernel,
                                 cl_device_id device,
                                 cl_uint work_dim,
                                 const size_t *local_work_size,
                                 cl_int *errcode_ret)
{
  cl_int errcode = CL_SUCCESS;
  cl_host_kernel_pocl hk = NULL;

  POCL_GOTO_ERROR_COND ((!IS_CL_OBJECT_VALID (kernel)), CL_INVALID_KERNEL);
  POCL_GOTO_ERROR_COND ((!IS_CL_OBJECT_VALID (device)), CL_INVALID_DEVICE);
  POCL_GOTO_ERROR_ON (
    (!pocl_device_is_associated_with_kernel (device, kernel)),
    CL_INVALID_DEVICE, "the kernel's context doesn't contain the device\n");

  /* Programs are built for the root device of a sub-device. */
  cl_device_id realdev = pocl_real_dev (device);
  POCL_GOTO_ERROR_ON (
    (realdev->ops->create_host_kernel == NULL), CL_INVALID_OPERATION,
    "device %s doesn't support cl_pocl_host_kernel\n", realdev->short_name);

  cl_program program = kernel->program;
  pocl_kernel_metadata_t *meta = kernel->meta;
  cl_uint dev_i = CL_UINT_MAX;
  for (cl_uint i = 0; i < program->num_devices; ++i)
    if (program->devices[i] == realdev)
      dev_i = i;
  POCL_GOTO_ERROR_ON ((dev_i == CL_UINT_MAX), CL_INVALID_PROGRAM_EXECUTABLE,
                      "the kernel was not built for this device\n");
  POCL_GOTO_ERROR_ON ((program->builtin_kernel_attributes != NULL),
                      CL_INVALID_KERNEL,
                      "built-in kernels can't be host kernels\n");

  POCL_GOTO_ERROR_COND (
    (work_dim < 1 || work_dim > realdev->max_work_item_dimensions),
    CL_INVALID_WORK_DIMENSION);
  POCL_GOTO_ERROR_COND ((local_work_size == NULL), CL_INVALID_WORK_GROUP_SIZE);

  size_t local_size[3] = { 1, 1, 1 };
  size_t wg_size = 1;
  for (cl_uint d = 0; d < work_dim; ++d)
    {
      local_size[d] = local_work_size[d];
      POCL_GOTO_ERROR_ON ((local_size[d] == 0
                           || local_size[d] > realdev->max_work_item_sizes[d]),
                          CL_INVALID_WORK_ITEM_SIZE,
                          "local size %zu in dimension %u is out of range\n",
                          local_size[d], d);
      POCL_GOTO_ERROR_ON ((mul_overflows (wg_size, local_size[d], &wg_size)),
                          CL_INVALID_WORK_GROUP_SIZE,
                          "the work-group size overflows\n");
    }
  size_t max_wg_size = meta->devices[dev_i].max_workgroup_size > 0
                         ? meta->devices[dev_i].max_workgroup_size
                         : realdev->max_work_group_size;
  POCL_GOTO_ERROR_ON ((wg_size > max_wg_size), CL_INVALID_WORK_GROUP_SIZE,
                      "work-group size %zu exceeds the kernel's maximum "
                      "%zu\n",
                      wg_size, max_wg_size);
  if (meta->reqd_wg_size[0] > 0)
    POCL_GOTO_ERROR_ON (
      (local_size[0] != meta->reqd_wg_size[0]
       || local_size[1] != meta->reqd_wg_size[1]
       || local_size[2] != meta->reqd_wg_size[2]),
      CL_INVALID_WORK_GROUP_SIZE,
      "the local size doesn't match the kernel's reqd_work_group_size\n");

  hk = calloc (1, sizeof (struct _cl_host_kernel_pocl));
  POCL_GOTO_ERROR_COND ((hk == NULL), CL_OUT_OF_HOST_MEMORY);
  POCL_INIT_OBJECT_NO_ICD (hk);
  /* The kernel keeps the program, its global variables and the module
     cache entry valid: OpenCL doesn't allow rebuilding a program that has
     kernels. */
  POname (clRetainKernel) (kernel);
  hk->kernel = kernel;
  hk->program = program;
  hk->device = realdev;
  hk->program_device_i = dev_i;
  hk->work_dim = work_dim;
  memcpy (hk->local_size, local_size, sizeof (local_size));
  hk->num_args = meta->num_args;
  hk->num_locals = meta->num_locals;

  hk->local_offsets
    = malloc (sizeof (size_t) * (meta->num_args + meta->num_locals + 1));
  POCL_GOTO_ERROR_COND ((hk->local_offsets == NULL), CL_OUT_OF_HOST_MEMORY);
  hk->arg_alignment_masks
    = calloc (meta->num_args + meta->num_locals + 1, sizeof (size_t));
  POCL_GOTO_ERROR_COND ((hk->arg_alignment_masks == NULL),
                        CL_OUT_OF_HOST_MEMORY);

  size_t offset = 0;
  for (cl_uint i = 0; i < meta->num_args; ++i)
    {
      struct pocl_argument_info *ai = &meta->arg_info[i];
      hk->local_offsets[i] = SIZE_MAX;
      if (ARG_IS_LOCAL (meta->arg_info[i]))
        {
          POCL_GOTO_ERROR_ON (
            (!kernel->dyn_arguments[i].is_set), CL_INVALID_KERNEL_ARGS,
            "the size of __local argument %u isn't set\n", i);
          hk->local_offsets[i] = offset;
          POCL_GOTO_ERROR_ON (
            (!reserve_local (&offset, kernel->dyn_arguments[i].size,
                             realdev->local_mem_size)),
            CL_OUT_OF_RESOURCES, "__local argument %u doesn't fit\n", i);
        }
      else
        {
          POCL_GOTO_ERROR_ON ((ai->type == POCL_ARG_TYPE_IMAGE
                               || ai->type == POCL_ARG_TYPE_SAMPLER
                               || ai->type == POCL_ARG_TYPE_PIPE),
                              CL_INVALID_KERNEL,
                              "argument %u is an image, sampler or pipe\n", i);
          /* The work-group function loads scalars and vectors with their
             preferred alignment, which OpenCL C makes their size rounded up
             to a power of two. The size of structs isn't known. */
          if (ai->type_size > 0)
            {
              size_t align = pocl_size_ceil2 (ai->type_size);
              if (align > MAX_EXTENDED_ALIGNMENT)
                align = MAX_EXTENDED_ALIGNMENT;
              hk->arg_alignment_masks[i] = align - 1;
            }
        }
    }
  for (cl_uint i = 0; i < meta->num_locals; ++i)
    {
      hk->local_offsets[meta->num_args + i] = offset;
      POCL_GOTO_ERROR_ON ((!reserve_local (&offset, meta->local_sizes[i],
                                           realdev->local_mem_size)),
                          CL_OUT_OF_RESOURCES,
                          "the kernel's __local variables don't fit\n");
    }
  hk->local_mem_size = offset;
  /* Kernels flush the printf buffer after every printf with
     ENABLE_PRINTF_IMMEDIATE_FLUSH, otherwise the run loop flushes it after
     every work-group that printed. */
  hk->printf_buffer_size = realdev->printf_buffer_size;
  hk->scratch_size = hk->local_mem_size + hk->printf_buffer_size;

  errcode = realdev->ops->create_host_kernel (realdev, kernel, hk);
  if (errcode != CL_SUCCESS)
    goto ERROR;

  if (errcode_ret)
    *errcode_ret = CL_SUCCESS;
  return hk;

ERROR:
  if (hk)
    free_host_kernel (hk);
  if (errcode_ret)
    *errcode_ret = errcode;
  return NULL;
}
POsym (clCreateHostKernelPOCL)

  CL_API_ENTRY cl_int CL_API_CALL
  POname (clGetHostKernelInfoPOCL) (cl_host_kernel_pocl host_kernel,
                                    cl_host_kernel_info_pocl param_name,
                                    size_t param_value_size,
                                    void *param_value,
                                    size_t *param_value_size_ret)
{
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (host_kernel)),
                          CL_INVALID_HOST_KERNEL_POCL);

  switch (param_name)
    {
    case CL_HOST_KERNEL_REFERENCE_COUNT_POCL:
      POCL_RETURN_GETINFO (cl_uint, (cl_uint)host_kernel->pocl_refcount);
    case CL_HOST_KERNEL_SCRATCH_SIZE_POCL:
      POCL_RETURN_GETINFO (size_t, host_kernel->scratch_size);
    case CL_HOST_KERNEL_SCRATCH_ALIGNMENT_POCL:
      POCL_RETURN_GETINFO (size_t, (size_t)MAX_EXTENDED_ALIGNMENT);
    default:
      return CL_INVALID_VALUE;
    }
}
POsym (clGetHostKernelInfoPOCL)

  CL_API_ENTRY cl_int CL_API_CALL
  POname (clRetainHostKernelPOCL) (cl_host_kernel_pocl host_kernel)
{
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (host_kernel)),
                          CL_INVALID_HOST_KERNEL_POCL);
  POCL_RETAIN_OBJECT (host_kernel);
  return CL_SUCCESS;
}
POsym (clRetainHostKernelPOCL)

  CL_API_ENTRY cl_int CL_API_CALL
  POname (clReleaseHostKernelPOCL) (cl_host_kernel_pocl host_kernel)
{
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (host_kernel)),
                          CL_INVALID_HOST_KERNEL_POCL);
  int new_refcount;
  POCL_RELEASE_OBJECT (host_kernel, new_refcount);
  if (new_refcount == 0)
    free_host_kernel (host_kernel);
  return CL_SUCCESS;
}
POsym (clReleaseHostKernelPOCL)

  CL_API_ENTRY cl_int CL_API_CALL
  POname (clRunHostKernelPOCL) (cl_host_kernel_pocl host_kernel,
                                const void *const *arg_values,
                                const size_t *num_groups,
                                const size_t *global_work_offset,
                                size_t first_group,
                                size_t group_count,
                                void *scratch)
{
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (host_kernel)),
                          CL_INVALID_HOST_KERNEL_POCL);
  POCL_RETURN_ERROR_COND ((num_groups == NULL), CL_INVALID_VALUE);
  POCL_RETURN_ERROR_COND ((arg_values == NULL && host_kernel->num_args > 0),
                          CL_INVALID_KERNEL_ARGS);
  POCL_RETURN_ERROR_COND (
    (((uintptr_t)scratch & (MAX_EXTENDED_ALIGNMENT - 1)) != 0),
    CL_INVALID_VALUE);
  POCL_RETURN_ERROR_COND ((scratch == NULL), CL_INVALID_VALUE);

  size_t total = 1;
  for (cl_uint d = 0; d < host_kernel->work_dim; ++d)
    {
      size_t global, extent;
      POCL_RETURN_ERROR_ON (
        (mul_overflows (num_groups[d], host_kernel->local_size[d], &global)
         || (global_work_offset != NULL
             && add_overflows (global, global_work_offset[d], &extent))
         || mul_overflows (total, num_groups[d], &total)),
        CL_INVALID_GLOBAL_WORK_SIZE, "the grid is too large\n");
    }
  POCL_RETURN_ERROR_ON (
    (first_group > total || group_count > total - first_group),
    CL_INVALID_VALUE, "work-groups [%zu, %zu) are outside the grid\n",
    first_group, first_group + group_count);
  if (group_count == 0)
    return CL_SUCCESS;

  return host_kernel->device->ops->run_host_kernel (
    host_kernel, arg_values, num_groups, global_work_offset, first_group,
    group_count, scratch);
}
POsym (clRunHostKernelPOCL)
