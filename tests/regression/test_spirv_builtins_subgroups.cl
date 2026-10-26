/* Sub-group kernels for test_spirv_builtins.c, built like
   test_spirv_builtins.cl to test_spirv_builtins_subgroups{32,64}.spv. */

kernel void
subgroups (global int *out)
{
  int gid = get_global_id (0);
  out[2 * gid] = sub_group_reduce_add (1) - (int)get_sub_group_size ();
  out[2 * gid + 1] = sub_group_broadcast (gid, 0)
                     - (gid - (int)get_sub_group_local_id ());
}
