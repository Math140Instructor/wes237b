#define TILE_WIDTH 16
#define KERNEL_SZ 7

__kernel void im2col(__global float *unrolled, __global float *x, const int B, const int C_in, const int H, const int W, const int K) {

#define x4d(batch, channel, row, col) x[(batch) * (C_in * H * W) + (channel) * (H * W) + (row) * W + (col)]

#define x_unroll_3d(batch, row, col) unrolled[((batch) * H_unroll + (row)) * W_unroll + (col)]

  // Output dimensions after valid convolution
  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

  // Dimensions of the im2col matrix
  const int H_unroll = C_in * K * K;
  const int W_unroll = H_out * W_out;

  const int unrolledCol = get_global_id(0);
  const int unrolledRow = get_global_id(1);
  const int batchIndex = get_global_id(2);

  // Protect against padded global work sizes
  if (unrolledCol >= W_unroll || unrolledRow >= H_unroll || batchIndex >= B) {
    return;
  }

  /*
   * Each row of the unrolled matrix:
   *
   * input channel
   * kernel row
   * kernel column
   *
   */

  const int inputChannel = unrolledRow / (K * K);
  const int kernelElement = unrolledRow % (K * K);
  const int kernelRow = kernelElement / K;
  const int kernelCol = kernelElement % K;

  /*
   * Each column of the unrolled matrix pixel location.
   */

  const int outputRow = unrolledCol / W_out;
  const int outputCol = unrolledCol % W_out;

  /*
   * Determine the right input pixel location of the unrolled matrix.
   */

  const int inputRow = outputRow + kernelRow;
  const int inputCol = outputCol + kernelCol;

  /*
   * Copy:
   * to
   * unrolled[batch][unrolledRow][unrolledCol]
   */

  x_unroll_3d(batchIndex, unrolledRow, unrolledCol) = x4d(batchIndex, inputChannel, inputRow, inputCol);

#undef x4d
#undef x_unroll_3d
}