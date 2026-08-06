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

  int rows, cols;
  rows = host_a.shape[0];
  cols = host_b.shape[1];

  output.shape[0] = 1;
  output.shape[1] = 1;
  output.data = (int *)malloc(sizeof(int) * rows * cols); // 3x3 => 1x9

  // Sum all elements of the array
  //@@ Modify the below code in the remaining demos
  int sum = 0;
  int arr = rows * cols;

  for (int i = 0; i < arr; i += 4) {
    if (i < arr)
      sum += host_a.data[i];
    if (i + 1 < arr)
      sum += host_a.data[i + 1];
    if (i + 2 < arr)
      sum += host_a.data[i + 2];
    if (i + 3 < arr)
      sum += host_a.data[i + 3];
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