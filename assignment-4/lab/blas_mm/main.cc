#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <clblast.h>

#include "device.h"
#include "kernel.h"
#include "matrix.h"

#include <chrono>

#define CHECK_ERR(err, msg)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            \
  if (err != CL_SUCCESS) {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             \
    fprintf(stderr, "%s failed: %d\n", msg, err);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      \
    exit(EXIT_FAILURE);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                \
  }

void CopyMatrix(const Matrix *input, float *output) {
  for (unsigned int i = 0; i < input->shape[0] * input->shape[1]; ++i) {
    output[i] = (float)input->data[i];
  }
}

void OpenCLMatrixMultiply(Matrix *input0, Matrix *input1, Matrix *result) {

  // Device input and output buffers
  cl_mem device_a, device_b, device_c;

  cl_int err;

  cl_device_id device_id; // device ID
  cl_context context;     // context
  cl_command_queue queue; // command queue

  // Find platforms and devices
  OclPlatformProp *platforms = NULL;
  cl_uint num_platforms;

  err = OclFindPlatforms((const OclPlatformProp **)&platforms, &num_platforms);
  CHECK_ERR(err, "OclFindPlatforms");

  err = OclGetDeviceWithFallback(&device_id, OCL_DEVICE_TYPE);
  CHECK_ERR(err, "OclGetDeviceWithFallback");

  // Create a context
  context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
  CHECK_ERR(err, "clCreateContext");

  // Create a command queue
#if __APPLE__
  queue = clCreateCommandQueue(context, device_id, 0, &err);
#else
  queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
#endif
  CHECK_ERR(err, "clCreateCommandQueueWithProperties");

  // Matrix dimensions:
  // A = NxP
  // B = PxM
  // C = NxM
  const size_t N = input0->shape[0];
  const size_t P = input0->shape[1];
  const size_t M = input1->shape[1];
  const size_t num_size_a = N * P, num_size_b = P * M, num_size_c = N * M;

  size_t B = 1;

  const size_t bytes_a = num_size_a * sizeof(float);
  const size_t bytes_b = num_size_b * sizeof(float);
  const size_t bytes_c = num_size_c * sizeof(float);

  float *h_A = (float *)malloc(bytes_a);
  float *h_B = (float *)malloc(bytes_b);
  float *h_C = (float *)malloc(bytes_c);

  CopyMatrix(input0, h_A);
  CopyMatrix(input1, h_B);

  std::vector<size_t> a_offsets = std::vector<size_t>(B, 0);
  std::vector<size_t> b_offsets = std::vector<size_t>(B, 0);
  std::vector<size_t> c_offsets = std::vector<size_t>(B, 0);

  std::vector<float> alphas = std::vector<float>(B, 1);
  std::vector<float> betas = std::vector<float>(B, 0);

  device_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes_a, NULL, &err);
  CHECK_ERR(err, "clCreateBuffer A");
  device_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes_b, NULL, &err);
  CHECK_ERR(err, "clCreateBuffer B");
  device_c = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes_c, NULL, &err);
  CHECK_ERR(err, "clCreateBuffer C");

  err = clEnqueueWriteBuffer(queue, device_a, CL_TRUE, 0, bytes_a, h_A, 0, NULL, NULL);
  CHECK_ERR(err, "clEnqueueWriteBuffer A");
  err = clEnqueueWriteBuffer(queue, device_b, CL_TRUE, 0, bytes_b, h_B, 0, NULL, NULL);
  CHECK_ERR(err, "clEnqueueWriteBuffer B");

  clblast::GemmBatched<float>(clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kNo, // no transposes
                              N, M, P,                                                                      //
                              alphas.data(),                                                                //
                              device_a, a_offsets.data(), P,                                                //
                              device_b, b_offsets.data(), M,                                                //
                              betas.data(),                                                                 //
                              device_c, c_offsets.data(), M,
                              B, // batch_count
                              &queue, nullptr);

  // const size_t candidates[] = {1, 1, 2, 4, 8, 16, 32};
  // for (size_t B : candidates) {
  //   std::vector<float> alphas(B, 1.0f);
  //   std::vector<float> betas(B, 0.0f);
  //   std::vector<size_t> a_offsets(B, 0);
  //   std::vector<size_t> b_offsets(B, 0);
  //   std::vector<size_t> c_offsets(B, 0);
  //   clFinish(queue);
  //   auto start = std::chrono::high_resolution_clock::now();
  //   clblast::GemmBatched<float>(clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kNo, N, M, P, alphas.data(), device_a, a_offsets.data(), P, device_b, b_offsets.data(), M, betas.data(), device_c, c_offsets.data(), M, B, &queue, nullptr);
  //   clFinish(queue);
  //   auto end = std::chrono::high_resolution_clock::now();
  //   double ms = std::chrono::duration<double, std::milli>(end - start).count();
  //   printf("N=%zu M=%zu P=%zu B=%zu time=%.4f ms\n", N, M, P, B, ms);
  // } /////

  //@@ Copy the GPU memory back to the CPU here
  err = clEnqueueReadBuffer(queue, device_c, CL_TRUE, 0, bytes_c, h_C, 0, NULL, NULL);
  CHECK_ERR(err, "clEnqueueReadBuffer C");

  // Copy back from h_C to result
  for (unsigned int i = 0; i < num_size_c; ++i) {
    result->data[i] = (int)h_C[i];
  }

  // Free GPU memory
  clReleaseMemObject(device_a);
  clReleaseMemObject(device_b);
  clReleaseMemObject(device_c);

  clblast::ClearCache();
  // Release the malloc'd memory
  free(h_A);
  free(h_B);
  free(h_C);
}

int main(int argc, char *argv[]) {
  if (argc != 5) {
    fprintf(stderr, "Usage: %s <input_file_0> <input_file_1> <answer_file> <output_file>\n", argv[0]);
    return -1;
  }

  const char *input_file_a = argv[1];
  const char *input_file_b = argv[2];
  const char *input_file_c = argv[3];
  const char *input_file_d = argv[4];

  // Host input and output vectors and sizes
  Matrix host_a, host_b, host_c, answer;

  cl_int err;

  err = LoadMatrix(input_file_a, &host_a);
  CHECK_ERR(err, "LoadMatrix");

  err = LoadMatrix(input_file_b, &host_b);
  CHECK_ERR(err, "LoadMatrix");

  err = LoadMatrix(input_file_c, &answer);
  CHECK_ERR(err, "LoadMatrix");

  int rows = host_a.shape[0], cols = host_b.shape[1];

  // Allocate the memory for the target.
  host_c.shape[0] = rows;
  host_c.shape[1] = cols;
  host_c.data = (int *)malloc(sizeof(int) * host_c.shape[0] * host_c.shape[1]);

  // Call your matrix multiply.
  OpenCLMatrixMultiply(&host_a, &host_b, &host_c);

  // // Call to print the matrix
  // PrintMatrix(&host_c);

  // Save the matrix
  SaveMatrix(input_file_d, &host_c);

  // Check the result of the matrix multiply
  CheckMatrix(&answer, &host_c);

  // Release host memory
  free(host_a.data);
  free(host_b.data);
  free(host_c.data);
  free(answer.data);

  return 0;
}
