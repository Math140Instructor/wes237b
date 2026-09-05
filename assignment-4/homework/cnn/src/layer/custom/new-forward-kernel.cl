#define TILE_WIDTH 16
#define KERNEL_SZ 7

// default implementation
__kernel void im2col(__global float *unrolled, __global float *x, const int B, const int C_in, const int H, const int W, const int K) {

#define x4d(i3, i2, i1, i0) x[(i3) * (C_in * H * W) + (i2) * (H * W) + (i1) * W + (i0)]

  // valid convolution output dimensions
  const int H_out = H - K + 1;
  const int W_out = W - K + 1;
  const int H_unroll = C_in * K * K;
  const int W_unroll = H_out * W_out;

#define x_unroll_3d(i2, i1, i0) unrolled[((i2) * H_unroll + (i1)) * W_unroll + (i0)]
  // compute one element of x_unroll
  int col_u = get_global_id(0);
  int row_u = get_global_id(1);
  int b = get_global_id(2);

  if (b < B && row_u < H_unroll && col_u < W_unroll) {

    /*
     * row_u:
     *   channel
     *   mask row
     *   mask column
     */
    int c = row_u / (K * K);
    int kernelIndex = row_u % (K * K);
    int maskRow = kernelIndex / K;
    int maskCol = kernelIndex % K;

    /*
     * col_u is one output convolution position
     * flattened in row major order
     */
    int row_o = col_u / W_out;
    int col_o = col_u % W_out;

    /*
     * Find the right input pixel
     */
    int row_i = row_o + maskRow;
    int col_i = col_o + maskCol;

    /*
     * Copy input pixel into unrolled matrix
     */
    x_unroll_3d(b, row_u, col_u) = x4d(b, c, row_i, col_i);
  }

#undef x4d
#undef x_unroll_3d
}