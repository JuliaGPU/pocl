===================================
OpenCL Extensions Supported by PoCL
===================================

PoCL supports a number of OpenCL extensions. The exact
list of extensions depends on the driver backend in use
as well as the exact options PoCL was built with.
Applications should always query available extensions
before attempting to use their functionality.

Full extension specifications can be found on:

https://www.khronos.org/registry/OpenCL/

Some highlights from the list of supported extensions:

cl_pocl_content_size
~~~~~~~~~~~~~~~~~~~~~~~

This extension provides a way to to indicate
a buffer which will hold the meaningful
bytes of another buffer, after kernel execution.

This allows the implementation to reduce the amount
of data copied when moving buffers between devices
e.g. when the data is compressed and its exact
length is not known ahead of time.


cl_pocl_host_kernel
~~~~~~~~~~~~~~~~~~~

A draft extension for CPU devices whose kernels run in the host process
(``cpu``, ``cpu-minimal``, ``cpu-tbb``). It lets an application run the
work-groups of a kernel on its own threads, synchronously and without command
queues, events or locks, so that very small launches cost little more than a
function call and the application's own scheduler (e.g. a language runtime's
thread pool) decides where work-groups run.

``clCreateHostKernelPOCL`` prepares a kernel for one device and local size,
and ``clRunHostKernelPOCL`` runs a range of the work-groups of a grid on the
calling thread, given the argument values and a scratch area for ``__local``
memory and printf. Work-groups may run in any order and concurrently, so they
must not wait on each other. See ``include/CL/cl_ext_pocl.h`` for the full
description.

cl_khr_command_buffer
~~~~~~~~~~~~~~~~~~~~~~~

This extension provides a way to record a sequence
of OpenCL commands that can be executed as a single
invocation. Command parameters are validated and
commands are prepared at command buffer recording
time, reducing the overhead of dispatching the sequence
and allowing drivers to optimize the scheduling of
commands within a buffer.

cl_ext_defined_builtin_kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The purpose of this extension is to provide a standardized
set of built-in kernels with well-defined semantics.
See :ref:`defined-built-in-kernels` for more details on
DBKs and PoCL's implementation of them.
