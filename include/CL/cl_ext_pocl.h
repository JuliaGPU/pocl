/*******************************************************************************
 * Copyright (c) 2021 Tampere University
 *               2023-2024 Pekka Jääskeläinen / Intel Finland Oy
 *
 * PoCL-specific proof-of-concept (draft) or finalized OpenCL extensions.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
 * KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
 * SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
 *    https://www.khronos.org/registry/
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 ******************************************************************************/

#include <CL/cl_ext.h>

#ifndef __CL_EXT_POCL_H
#define __CL_EXT_POCL_H

#ifdef __cplusplus
extern "C"
{
#endif

/* cl_pocl_content_size should be defined in CL/cl_ext.h; however,
 * if we PoCL is built against the system headers, it's possible
 * that they have an outdated version of CL/cl_ext.h.
 * In that case, add the extension here.
 */

#ifndef cl_pocl_content_size

#define cl_pocl_content_size 1

extern CL_API_ENTRY cl_int CL_API_CALL
clSetContentSizeBufferPoCL(
    cl_mem    buffer,
    cl_mem    content_size_buffer) CL_API_SUFFIX__VERSION_1_2;

typedef CL_API_ENTRY cl_int
(CL_API_CALL *clSetContentSizeBufferPoCL_fn)(
    cl_mem    buffer,
    cl_mem    content_size_buffer) CL_API_SUFFIX__VERSION_1_2;

#endif

#define CL_DEVICE_REMOTE_TRAFFIC_STATS_POCL 0x4501

#define CL_DEVICE_REMOTE_SERVER_IP_POCL 0x4503

#define CL_DEVICE_REMOTE_SERVER_PORT_POCL 0x4504

  /***********************************
   * cl_pocl_svm_rect +
   * cl_pocl_command_buffer_svm +
   * cl_pocl_command_buffer_host_buffer
   * extensions
   ************************************/

#ifdef cl_khr_command_buffer

// SVM memory command-buffer functions (clCommandSVMMemcpyPOCL etc)
#define cl_pocl_command_buffer_svm 1

// cl_mem & host command-buffer functions (clCommandReadBuffer etc)
#define cl_pocl_command_buffer_host_buffer 1

// these are separate from command-buffers
// clEnqueueSVMMemFillRectPOCL, clEnqueueSVMMemcpyRectPOCL
#define cl_pocl_svm_rect 1

/****************************************************/

/* cl_device_command_buffer_capabilities_khr - bitfield */
#define CL_COMMAND_BUFFER_CAPABILITY_PROFILING_POCL  (1 << 8)

/* cl_command_buffer_flags_khr */
#define CL_COMMAND_BUFFER_PROFILING_POCL              (1 << 8)

/* cl_command_buffer_info_khr */
#define CL_COMMAND_BUFFER_INFO_PROFILING_POCL                     0x1299

/* cl_command_type */
/* To be used by clGetEventInfo: */
/* TODO use values from an assigned range */
#define CL_COMMAND_SVM_MEMCPY_RECT_POCL                       0x1210
#define CL_COMMAND_SVM_MEMFILL_RECT_POCL                      0x1211

typedef cl_int (CL_API_CALL *
clCommandSVMMemcpyPOCL_fn)(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    void *dst_ptr,
    const void *src_ptr,
    size_t size,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

typedef cl_int (CL_API_CALL *
clCommandSVMMemcpyRectPOCL_fn)(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    void *dst_ptr,
    const void *src_ptr,
    const size_t *dst_origin,
    const size_t *src_origin,
    const size_t *region,
    size_t dst_row_pitch,
    size_t dst_slice_pitch,
    size_t src_row_pitch,
    size_t src_slice_pitch,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

typedef cl_int (CL_API_CALL *
clCommandSVMMemfillPOCL_fn)(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    void *svm_ptr,
    size_t size,
    const void *pattern,
    size_t pattern_size,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);


typedef cl_int (CL_API_CALL *
clCommandSVMMemfillRectPOCL_fn)(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    void *svm_ptr,
    const size_t *origin,
    const size_t *region,
    size_t row_pitch,
    size_t slice_pitch,
    const void *pattern,
    size_t pattern_size,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);







typedef cl_int (CL_API_CALL *
clCommandReadBufferPOCL_fn)(cl_command_buffer_khr command_buffer,
                        cl_command_queue command_queue,
                        cl_mem buffer,
                        size_t offset,
                        size_t size,
                        void *ptr,
                        cl_uint num_sync_points_in_wait_list,
                        const cl_sync_point_khr* sync_point_wait_list,
                        cl_sync_point_khr* sync_point,
                        cl_mutable_command_khr* mutable_handle);

typedef cl_int (CL_API_CALL *
clCommandReadBufferRectPOCL_fn)(cl_command_buffer_khr command_buffer,
                            cl_command_queue command_queue,
                            cl_mem buffer,
                            const size_t *buffer_origin,
                            const size_t *host_origin,
                            const size_t *region,
                            size_t buffer_row_pitch,
                            size_t buffer_slice_pitch,
                            size_t host_row_pitch,
                            size_t host_slice_pitch,
                            void *ptr,
                            cl_uint num_sync_points_in_wait_list,
                            const cl_sync_point_khr* sync_point_wait_list,
                            cl_sync_point_khr* sync_point,
                            cl_mutable_command_khr* mutable_handle);

typedef cl_int (CL_API_CALL *
clCommandReadImagePOCL_fn)(cl_command_buffer_khr command_buffer,
                       cl_command_queue command_queue,
                       cl_mem               image,
                       const size_t *       origin, /* [3] */
                       const size_t *       region, /* [3] */
                       size_t               row_pitch,
                       size_t               slice_pitch,
                       void *               ptr,
                       cl_uint num_sync_points_in_wait_list,
                       const cl_sync_point_khr* sync_point_wait_list,
                       cl_sync_point_khr* sync_point,
                       cl_mutable_command_khr* mutable_handle);

typedef cl_int (CL_API_CALL *
clCommandWriteBufferPOCL_fn)(cl_command_buffer_khr command_buffer,
                         cl_command_queue command_queue,
                         cl_mem buffer,
                         size_t offset,
                         size_t size,
                         const void *ptr,
                         cl_uint num_sync_points_in_wait_list,
                         const cl_sync_point_khr* sync_point_wait_list,
                         cl_sync_point_khr* sync_point,
                         cl_mutable_command_khr* mutable_handle);

typedef cl_int (CL_API_CALL *
clCommandWriteBufferRectPOCL_fn)(cl_command_buffer_khr command_buffer,
                             cl_command_queue command_queue,
                             cl_mem buffer,
                             const size_t *buffer_origin,
                             const size_t *host_origin,
                             const size_t *region,
                             size_t buffer_row_pitch,
                             size_t buffer_slice_pitch,
                             size_t host_row_pitch,
                             size_t host_slice_pitch,
                             const void *ptr,
                             cl_uint num_sync_points_in_wait_list,
                             const cl_sync_point_khr* sync_point_wait_list,
                             cl_sync_point_khr* sync_point,
                             cl_mutable_command_khr* mutable_handle);

typedef cl_int (CL_API_CALL *
clCommandWriteImagePOCL_fn)(cl_command_buffer_khr command_buffer,
                        cl_command_queue    command_queue,
                        cl_mem              image,
                        const size_t *      origin, /*[3]*/
                        const size_t *      region, /*[3]*/
                        size_t              row_pitch,
                        size_t              slice_pitch,
                        const void *        ptr,
                        cl_uint num_sync_points_in_wait_list,
                        const cl_sync_point_khr* sync_point_wait_list,
                        cl_sync_point_khr* sync_point,
                        cl_mutable_command_khr* mutable_handle);

typedef cl_int (CL_API_CALL *
clEnqueueSVMMemcpyRectPOCL_fn) (cl_command_queue command_queue,
                            cl_bool blocking,
                            void *dst_ptr,
                            const void *src_ptr,
                            const size_t *dst_origin,
                            const size_t *src_origin,
                            const size_t *region,
                            size_t dst_row_pitch,
                            size_t dst_slice_pitch,
                            size_t src_row_pitch,
                            size_t src_slice_pitch,
                            cl_uint num_events_in_wait_list,
                            const cl_event *event_wait_list,
                            cl_event *event);

typedef cl_int (CL_API_CALL *
clEnqueueSVMMemFillRectPOCL_fn) (cl_command_queue  command_queue,
                             void *            svm_ptr,
                             const size_t *    origin,
                             const size_t *    region,
                             size_t            row_pitch,
                             size_t            slice_pitch,
                             const void *      pattern,
                             size_t            pattern_size,
                             size_t            size,
                             cl_uint           num_events_in_wait_list,
                             const cl_event *  event_wait_list,
                             cl_event *        event);


#ifndef CL_NO_PROTOTYPES

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandSVMMemcpyPOCL(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    void *dst_ptr,
    const void *src_ptr,
    size_t size,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandSVMMemcpyRectPOCL(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    void *dst_ptr,
    const void *src_ptr,
    const size_t *dst_origin,
    const size_t *src_origin,
    const size_t *region,
    size_t dst_row_pitch,
    size_t dst_slice_pitch,
    size_t src_row_pitch,
    size_t src_slice_pitch,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandSVMMemfillPOCL(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    void *svm_ptr,
    size_t size,
    const void *pattern,
    size_t pattern_size,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandSVMMemfillRectPOCL(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    void *svm_ptr,
    const size_t *origin,
    const size_t *region,
    size_t row_pitch,
    size_t slice_pitch,
    const void *pattern,
    size_t pattern_size,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);




extern CL_API_ENTRY cl_int CL_API_CALL
clCommandReadBufferPOCL(cl_command_buffer_khr command_buffer,
                        cl_command_queue command_queue,
                        cl_mem buffer,
                        size_t offset,
                        size_t size,
                        void *ptr,
                        cl_uint num_sync_points_in_wait_list,
                        const cl_sync_point_khr* sync_point_wait_list,
                        cl_sync_point_khr* sync_point,
                        cl_mutable_command_khr* mutable_handle);

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandReadBufferRectPOCL(cl_command_buffer_khr command_buffer,
                            cl_command_queue command_queue,
                            cl_mem buffer,
                            const size_t *buffer_origin,
                            const size_t *host_origin,
                            const size_t *region,
                            size_t buffer_row_pitch,
                            size_t buffer_slice_pitch,
                            size_t host_row_pitch,
                            size_t host_slice_pitch,
                            void *ptr,
                            cl_uint num_sync_points_in_wait_list,
                            const cl_sync_point_khr* sync_point_wait_list,
                            cl_sync_point_khr* sync_point,
                            cl_mutable_command_khr* mutable_handle);

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandReadImagePOCL(cl_command_buffer_khr command_buffer,
                       cl_command_queue command_queue,
                       cl_mem               image,
                       const size_t *       origin, /* [3] */
                       const size_t *       region, /* [3] */
                       size_t               row_pitch,
                       size_t               slice_pitch,
                       void *               ptr,
                       cl_uint num_sync_points_in_wait_list,
                       const cl_sync_point_khr* sync_point_wait_list,
                       cl_sync_point_khr* sync_point,
                       cl_mutable_command_khr* mutable_handle);

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandWriteBufferPOCL(cl_command_buffer_khr command_buffer,
                         cl_command_queue command_queue,
                         cl_mem buffer,
                         size_t offset,
                         size_t size,
                         const void *ptr,
                         cl_uint num_sync_points_in_wait_list,
                         const cl_sync_point_khr* sync_point_wait_list,
                         cl_sync_point_khr* sync_point,
                         cl_mutable_command_khr* mutable_handle);

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandWriteBufferRectPOCL(cl_command_buffer_khr command_buffer,
                             cl_command_queue command_queue,
                             cl_mem buffer,
                             const size_t *buffer_origin,
                             const size_t *host_origin,
                             const size_t *region,
                             size_t buffer_row_pitch,
                             size_t buffer_slice_pitch,
                             size_t host_row_pitch,
                             size_t host_slice_pitch,
                             const void *ptr,
                             cl_uint num_sync_points_in_wait_list,
                             const cl_sync_point_khr* sync_point_wait_list,
                             cl_sync_point_khr* sync_point,
                             cl_mutable_command_khr* mutable_handle);

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandWriteImagePOCL(cl_command_buffer_khr command_buffer,
                        cl_command_queue    command_queue,
                        cl_mem              image,
                        const size_t *      origin, /*[3]*/
                        const size_t *      region, /*[3]*/
                        size_t              row_pitch,
                        size_t              slice_pitch,
                        const void *        ptr,
                        cl_uint num_sync_points_in_wait_list,
                        const cl_sync_point_khr* sync_point_wait_list,
                        cl_sync_point_khr* sync_point,
                        cl_mutable_command_khr* mutable_handle);

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueSVMMemcpyRectPOCL (cl_command_queue command_queue,
                            cl_bool blocking,
                            void *dst_ptr,
                            const void *src_ptr,
                            const size_t *dst_origin,
                            const size_t *src_origin,
                            const size_t *region,
                            size_t dst_row_pitch,
                            size_t dst_slice_pitch,
                            size_t src_row_pitch,
                            size_t src_slice_pitch,
                            cl_uint num_events_in_wait_list,
                            const cl_event *event_wait_list,
                            cl_event *event);

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueSVMMemFillRectPOCL (cl_command_queue  command_queue,
                             void *            svm_ptr,
                             const size_t *    origin,
                             const size_t *    region,
                             size_t            row_pitch,
                             size_t            slice_pitch,
                             const void *      pattern,
                             size_t            pattern_size,
                             size_t            size,
                             cl_uint           num_events_in_wait_list,
                             const cl_event *  event_wait_list,
                             cl_event *        event);

extern CL_API_ENTRY cl_int CL_API_CALL clSetKernelArgDevicePointerEXT (
    cl_kernel kernel, cl_uint arg_index, cl_mem_device_address_ext dev_addr);

#endif // CL_NO_PROTOTYPES

#endif // cl_khr_command_buffer

/* cl_pocl_host_kernel (draft): run work-groups of a kernel synchronously on
 * the calling thread, without command queues or events, so that the
 * application can schedule them on its own threads. Only CPU devices whose
 * kernels execute in the host process support it.
 *
 * clCreateHostKernelPOCL prepares a kernel for one local size on one device:
 * it compiles (or looks up) the work-group function, runs the initializers of
 * program-scope variables and records the sizes of the __local arguments
 * (which must have been set with clSetKernelArg). The handle doesn't depend on
 * the kernel's argument values. It retains the kernel, so the application may
 * release the kernel and program, but can't rebuild the program, while the
 * handle is alive.
 *
 * clRunHostKernelPOCL runs the work-groups with linear indices
 * [first_group, first_group + group_count) of a grid of num_groups groups
 * (dimension 0 varying fastest) on the calling thread and returns when they
 * are done. It takes no locks and doesn't allocate. arg_values[i] points to
 * the value of argument i, as clSetKernelArg would take it, except that
 * pointer arguments take the address the kernel uses (an SVM, USM or device
 * address) and that the value of a struct must be aligned to the struct's
 * alignment (values of other types are copied if they aren't aligned);
 * entries for __local arguments are ignored. scratch must point to
 * CL_HOST_KERNEL_SCRATCH_SIZE_POCL bytes aligned to
 * CL_HOST_KERNEL_SCRATCH_ALIGNMENT_POCL that no concurrent call uses; it
 * holds the __local memory and the printf buffer. On x86 and AArch64, the
 * kernel runs in its own floating-point environment (round to nearest,
 * denormal handling per the build options, all exceptions masked). The
 * caller's control modes and the exception flags it had raised are preserved;
 * as with a function call, the call may raise further flags, which ones being
 * unspecified.
 *
 * Work-groups may be run in any order, from any number of threads and
 * concurrently, so they must not wait on each other. Only kernels without
 * image, sampler or pipe arguments are supported. The call returns
 * CL_OUT_OF_RESOURCES, after finishing its range, if a work-group reported a
 * failure, which happens in the same cases (depending on the work-item
 * handler) as a command queue launch of the kernel failing.
 *
 * The enum values below are provisional and not registered with Khronos. */

#ifndef cl_pocl_host_kernel
#define cl_pocl_host_kernel 1

#define CL_POCL_HOST_KERNEL_EXTENSION_NAME "cl_pocl_host_kernel"

typedef struct _cl_host_kernel_pocl *cl_host_kernel_pocl;
typedef cl_uint cl_host_kernel_info_pocl;

/* cl_host_kernel_info_pocl */
#define CL_HOST_KERNEL_REFERENCE_COUNT_POCL 0x4510
#define CL_HOST_KERNEL_SCRATCH_SIZE_POCL 0x4511
#define CL_HOST_KERNEL_SCRATCH_ALIGNMENT_POCL 0x4512

/* error codes */
#define CL_INVALID_HOST_KERNEL_POCL -1200

typedef cl_host_kernel_pocl (CL_API_CALL *clCreateHostKernelPOCL_fn) (
    cl_kernel kernel, cl_device_id device, cl_uint work_dim,
    const size_t *local_work_size, cl_int *errcode_ret);

typedef cl_int (CL_API_CALL *clGetHostKernelInfoPOCL_fn) (
    cl_host_kernel_pocl host_kernel, cl_host_kernel_info_pocl param_name,
    size_t param_value_size, void *param_value,
    size_t *param_value_size_ret);

typedef cl_int (CL_API_CALL *clRetainHostKernelPOCL_fn) (
    cl_host_kernel_pocl host_kernel);

typedef cl_int (CL_API_CALL *clReleaseHostKernelPOCL_fn) (
    cl_host_kernel_pocl host_kernel);

typedef cl_int (CL_API_CALL *clRunHostKernelPOCL_fn) (
    cl_host_kernel_pocl host_kernel, const void *const *arg_values,
    const size_t *num_groups, const size_t *global_work_offset,
    size_t first_group, size_t group_count, void *scratch);

#ifndef CL_NO_PROTOTYPES

extern CL_API_ENTRY cl_host_kernel_pocl CL_API_CALL clCreateHostKernelPOCL (
    cl_kernel kernel, cl_device_id device, cl_uint work_dim,
    const size_t *local_work_size, cl_int *errcode_ret);

extern CL_API_ENTRY cl_int CL_API_CALL clGetHostKernelInfoPOCL (
    cl_host_kernel_pocl host_kernel, cl_host_kernel_info_pocl param_name,
    size_t param_value_size, void *param_value,
    size_t *param_value_size_ret);

extern CL_API_ENTRY cl_int CL_API_CALL
clRetainHostKernelPOCL (cl_host_kernel_pocl host_kernel);

extern CL_API_ENTRY cl_int CL_API_CALL
clReleaseHostKernelPOCL (cl_host_kernel_pocl host_kernel);

extern CL_API_ENTRY cl_int CL_API_CALL clRunHostKernelPOCL (
    cl_host_kernel_pocl host_kernel, const void *const *arg_values,
    const size_t *num_groups, const size_t *global_work_offset,
    size_t first_group, size_t group_count, void *scratch);

#endif /* CL_NO_PROTOTYPES */

#endif /* cl_pocl_host_kernel */

#ifdef __cplusplus
}
#endif

#endif /* __CL_EXT_POCL_H */
