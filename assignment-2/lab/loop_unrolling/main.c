#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"

#define CHECK_ERR(err, msg)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            \
  if (err != CL_SUCCESS) {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             \
    fprintf(stderr, "%s failed: %d\n", msg, err);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      \
    exit(EXIT_FAILURE);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                \
  }

int main(int argc, char *argv[]) {
  if (argc != 4) {
    fprintf(stderr, "Usage: %s <input_file_0> <answer_file> <output_file>\n", argv[0]);
    return -1;
  }

  const char *input_file_a = argv[1];
  const char *input_file_b = argv[2];
  const char *output_file = argv[3];

  // Host input and output vectors and sizes
  Matrix host_a, host_b, output;

  cl_int err;

  err = LoadMatrix(input_file_a, &host_a);
  CHECK_ERR(err, "LoadMatrix");

  err = LoadMatrix(input_file_b, &host_b);
  CHECK_ERR(err, "LoadMatrix");

  // A: NxP
  // B: PxM
  if (host_a.shape[1] != host_b.shape[0])
    return -2;

  int rows, cols;
  rows = host_a.shape[0];
  cols = host_b.shape[1];

  output.shape[0] = 1;
  output.shape[1] = 1;
  output.data = (int *)malloc(sizeof(int) * rows * cols); // 3x3 => 1x9

  // Sum all elements of the array
  //@@ Modify the below code in the remaining demos
  int sum = 0, sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
  int arr = rows * cols;
  int i = 0;
  for (i = 0; i + 3 < arr; i += 4) {
    sum0 += host_a.data[i];
    sum1 += host_a.data[i + 1];
    sum2 += host_a.data[i + 2];
    sum3 += host_a.data[i + 3];
  }
  sum = sum0 + sum1 + sum2 + sum3;
  for (; i < arr; ++i) {
    sum += host_a.data[i];
  }

  printf("sum: %d == %d\n", sum, host_b.data[0]);

  output.data[0] = sum;
  err = CheckMatrix(&host_b, &output);
  CHECK_ERR(err, "CheckMatrix");
  SaveMatrix(output_file, &output);

  // Release host memory
  free(host_a.data);
  free(host_b.data);
  free(output.data);

  return 0;
}