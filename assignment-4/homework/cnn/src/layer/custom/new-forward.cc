#include <cmath>
#include <iostream>
#include <vector>

#include <clblast.h>

#include "device.h"
#include "kernel.h"

#include "opencl-new-forward.h"

#define CHECK_ERR(err, msg)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            \
  if (err != CL_SUCCESS) {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             \
    fprintf(stderr, "%s failed: %d.\n", msg, err);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     \
    exit(EXIT_FAILURE);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                \
  }

void OpenCLInterface::conv_forward_gemm_opencl_prolog(const float *host_y, const float *host_x, const float *host_k, cl_mem *device_y, cl_mem *device_x, cl_mem *device_k, cl_mem *device_x_unroll, const int B, const int M, const int C, const int H, const int W, const int K) {
  // Output dimensions
  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

  // Number of elements in each buffer
  const size_t x_size = (size_t)B * C * H * W;
  const size_t k_size = (size_t)M * C * K * K;
  const size_t y_size = (size_t)B * M * H_out * W_out;

  const size_t x_unroll_size = (size_t)B * C * K * K * H_out * W_out;

  cl_int err;

  //@@ Allocate GPU memory here (don't forget batch sizes!)

  *device_x = clCreateBuffer(opencl->context, CL_MEM_READ_ONLY, x_size * sizeof(float), NULL, &err);
  CHECK_ERR(err, "clCreateBuffer device_x");

  *device_k = clCreateBuffer(opencl->context, CL_MEM_READ_ONLY, k_size * sizeof(float), NULL, &err);
  CHECK_ERR(err, "clCreateBuffer device_k");

  *device_x_unroll = clCreateBuffer(opencl->context, CL_MEM_READ_WRITE, x_unroll_size * sizeof(float), NULL, &err);
  CHECK_ERR(err, "clCreateBuffer device_x_unroll");

  *device_y = clCreateBuffer(opencl->context, CL_MEM_READ_WRITE, y_size * sizeof(float), NULL, &err);
  CHECK_ERR(err, "clCreateBuffer device_y");

  //@@ Copy memory to the GPU here

  err = clEnqueueWriteBuffer(opencl->queue, *device_x, CL_TRUE, 0, x_size * sizeof(float), host_x, 0, NULL, NULL);
  CHECK_ERR(err, "clEnqueueWriteBuffer device_x");

  err = clEnqueueWriteBuffer(opencl->queue, *device_k, CL_TRUE, 0, k_size * sizeof(float), host_k, 0, NULL, NULL);
  CHECK_ERR(err, "clEnqueueWriteBuffer device_k");
}

void OpenCLInterface::conv_forward_gemm_opencl(cl_mem device_y, const cl_mem device_x, const cl_mem device_k, const cl_mem device_x_unroll, const int B, const int M, const int C, const int H, const int W, const int K) {
  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

  const int H_unroll = C * K * K;
  const int W_unroll = H_out * W_out;

  cl_int err;

  //@@ ====== Start im2col =====

  // Set im2col arguments
  err = clSetKernelArg(opencl->im2col_kernel, 0, sizeof(cl_mem), &device_x_unroll);
  CHECK_ERR(err, "clSetKernelArg 0");

  err = clSetKernelArg(opencl->im2col_kernel, 1, sizeof(cl_mem), &device_x);
  CHECK_ERR(err, "clSetKernelArg 1");

  err = clSetKernelArg(opencl->im2col_kernel, 2, sizeof(int), &B);
  CHECK_ERR(err, "clSetKernelArg 2");

  err = clSetKernelArg(opencl->im2col_kernel, 3, sizeof(int), &C);
  CHECK_ERR(err, "clSetKernelArg 3");

  err = clSetKernelArg(opencl->im2col_kernel, 4, sizeof(int), &H);
  CHECK_ERR(err, "clSetKernelArg 4");

  err = clSetKernelArg(opencl->im2col_kernel, 5, sizeof(int), &W);
  CHECK_ERR(err, "clSetKernelArg 5");

  err = clSetKernelArg(opencl->im2col_kernel, 6, sizeof(int), &K);
  CHECK_ERR(err, "clSetKernelArg 6");

  // @@ define local and global work sizes
  const size_t tile = 16;

  size_t localWorkSize[3] = {tile, tile, 1};
  size_t globalWorkSize[3] = {((W_unroll + tile - 1) / tile) * tile, ((H_unroll + tile - 1) / tile) * tile, (size_t)B};

  //@@ Launch the im2col kernel here

  err = clEnqueueNDRangeKernel(opencl->queue, opencl->im2col_kernel, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
  CHECK_ERR(err, "clEnqueueNDRangeKernel im2col");

  //@@ ====== End im2col =====
  //@@ ====== Start gemm =====

  /*
   * Matrix dimensions:
   *
   * A = device_k
   *     M x (C*K*K)
   *
   * B = device_x_unroll
   *     (C*K*K) x (H_out*W_out)
   *
   * C = device_y
   *     M x (H_out*W_out)
   */

  const size_t gemm_m = M;
  const size_t gemm_n = H_out * W_out;
  const size_t gemm_k = C * K * K;

  // One alpha/beta per batch
  std::vector<float> alphas(B, 1.0f);
  std::vector<float> betas(B, 0.0f);

  // One offset per batch
  std::vector<size_t> a_offsets(B);
  std::vector<size_t> b_offsets(B);
  std::vector<size_t> c_offsets(B);

  for (int b = 0; b < B; b++) {
    /*
     * The same convolution mask is used
     * for every batch element.
     */
    a_offsets[b] = 0;

    /*
     * Each batch has its own x_unroll matrix.
     *
     * Matrix size:
     * gemm_k * gemm_n
     */
    b_offsets[b] = (size_t)b * gemm_k * gemm_n;

    /*
     * Each batch has its own output matrix.
     *
     * Matrix size:
     * gemm_m * gemm_n
     */
    c_offsets[b] = (size_t)b * gemm_m * gemm_n;
  }

  /*
   * Row-major leading dimensions:
   *
   * A columns = gemm_k
   * B columns = gemm_n
   * C columns = gemm_n
   */
  const size_t lda = gemm_k;
  const size_t ldb = gemm_n;
  const size_t ldc = gemm_n;

  // @@ Call clblast::GemmBatched here

  clblast::StatusCode status = clblast::GemmBatched<float>(clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kNo,

                                                           gemm_m, gemm_n, gemm_k,

                                                           alphas.data(),

                                                           device_k, a_offsets.data(), lda,

                                                           device_x_unroll, b_offsets.data(), ldb,

                                                           betas.data(),

                                                           device_y, c_offsets.data(), ldc,

                                                           (size_t)B,

                                                           &opencl->queue, NULL);

  if (status != clblast::StatusCode::kSuccess) {
    fprintf(stderr, "clblast::GemmBatched failed: %d\n", static_cast<int>(status));

    exit(EXIT_FAILURE);
  }
 
  clblast::ClearCache();

  //@@ ====== End gemm =====
}

void OpenCLInterface::conv_forward_gemm_opencl_epilog(float *host_y, cl_mem device_y, cl_mem device_x, cl_mem device_k, cl_mem device_x_unroll, const int B, const int M, const int C, const int H, const int W, const int K) {
  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

  const size_t y_size = (size_t)B * M * H_out * W_out;

  //@@ Copy the output back to host

  cl_int err = clEnqueueReadBuffer(opencl->queue, device_y, CL_TRUE, 0, y_size * sizeof(float), host_y, 0, NULL, NULL);

  CHECK_ERR(err, "clEnqueueReadBuffer device_y");

  //@@ Free the GPU memory here

  clReleaseMemObject(device_y);
  clReleaseMemObject(device_x);
  clReleaseMemObject(device_k);
  clReleaseMemObject(device_x_unroll);
}