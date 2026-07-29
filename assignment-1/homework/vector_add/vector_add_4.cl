__kernel void vectorAdd(__global const int *a, __global const int *b,
                        __global const int *c, __global const int *d,
                        __global int *result, const unsigned int size)
{
  //@@ Insert code to implement vector addition here
  size_t id = get_global_id(0);
  result[id] = a[id] + b[id] + c[id] + d[id];
}
