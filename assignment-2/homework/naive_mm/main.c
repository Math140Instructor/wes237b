#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"

#define CHECK_ERR(err, msg)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            \
  if (err != CL_SUCCESS) {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             \
    fprintf(stderr, "%s failed: %d\n", msg, err);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      \
    exit(EXIT_FAILURE);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                \
  }

void NaiveMatrixMultiply(Matrix *input0, Matrix *input1, Matrix *result) {
  // NxP PxM
  // NxM
  // [1 2 3] [7 8]
  // [4 5 6] [9 10]
  //         [11 12]
  // 2x3 3x2 = 2x2
  //@@ Insert code to implement naive matrix multiply here
  int N = input0->shape[0], P = input0->shape[1], PP = input1->shape[0], M = input1->shape[1];
  if (P != PP) {
    return;
  }

  for (int r = 0; r < N; ++r) {
    for (int col = 0; col < M; ++col) {
      int sum = 0;
      for (int p = 0; p < P; ++p) {
        sum += input0->data[r * P + p] * input1->data[p * M + col];
      }
      result->data[r * M + col] = sum;
    }
  }
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

  // NxP , PXM
  int rows = host_a.shape[0], cols = host_b.shape[1];
  //@@ Update these values for the output rows and cols of the output
  //@@ Do not use the results from the answer matrix

  // Allocate the memory for the target.
  host_c.shape[0] = rows;
  host_c.shape[1] = cols;
  host_c.data = (int *)malloc(sizeof(int) * host_c.shape[0] * host_c.shape[1]);

  // Call your matrix multiply.
  NaiveMatrixMultiply(&host_a, &host_b, &host_c);

  // // Call to print the matrix
  // PrintMatrix(&host_c);

  // Save the matrix
  SaveMatrix(input_file_d, &host_c);

  // Check the result of the matrix multiply
  err = CheckMatrix(&answer, &host_c);
  CHECK_ERR(err, "CheckMatrix");

  // Release host memory
  free(host_a.data);
  free(host_b.data);
  free(host_c.data);
  free(answer.data);

  return 0;
}