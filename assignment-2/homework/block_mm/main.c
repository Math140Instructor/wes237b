#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"

#define CHECK_ERR(err, msg)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            \
  if (err != CL_SUCCESS) {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             \
    fprintf(stderr, "%s failed: %d\n", msg, err);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      \
    exit(EXIT_FAILURE);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                \
  }

#define BLOCK_SIZE 32

/**
 * 32x32 block sizes
 */
void BlockMatrixMultiply(Matrix *input0, Matrix *input1, Matrix *result) {
  // input0: N x P
  // input1: P x M
  // result: N x M
  int N = input0->shape[0];
  int P = input0->shape[1];
  int PP = input1->shape[0];
  int M = input1->shape[1];

  if (P != PP) {
    return;
  }

  // Initialize result matrix
  for (int r = 0; r < N; ++r) {
    for (int col = 0; col < M; ++col) {
      result->data[r * M + col] = 0;
    }
  }

  /* Move through the matrices one block at a time*/

  for (int rBlock = 0; rBlock < N; rBlock += BLOCK_SIZE) { // partition row into block chunks
    int rEnd = rBlock + BLOCK_SIZE;
    if (rEnd > N) {
      rEnd = N;
    }

    for (int colBlock = 0; colBlock < M; colBlock += BLOCK_SIZE) { // partition column into block chunks
      int colEnd = colBlock + BLOCK_SIZE;
      if (colEnd > M) {
        colEnd = M;
      }

      for (int pBlock = 0; pBlock < P; pBlock += BLOCK_SIZE) {
        int pEnd = pBlock + BLOCK_SIZE;
        if (pEnd > P) {
          pEnd = P;
        }

        // Multiply the current blocks
        for (int r = rBlock; r < rEnd; ++r) {
          for (int p = pBlock; p < pEnd; ++p) {
            int aValue = input0->data[r * P + p];
            for (int col = colBlock; col < colEnd; ++col) {
              result->data[r * M + col] += aValue * input1->data[p * M + col];
            }
          }
        }
      }
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

  // Allocate the memory for the target.
  host_c.shape[0] = rows;
  host_c.shape[1] = cols;
  host_c.data = (int *)malloc(sizeof(int) * host_c.shape[0] * host_c.shape[1]);

  // Call your matrix multiply.
  BlockMatrixMultiply(&host_a, &host_b, &host_c);

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