__kernel void matrixMultiply(__global const int *A, __global const int *B, __global int *C, const unsigned int numARows, const unsigned int numAColumns, const unsigned int numBRows, const unsigned int numBColumns, const unsigned int numCRows, const unsigned int numCColumns) {
  //@@ Compute C = AB
  const unsigned int row = get_global_id(0);
  const unsigned int col = get_global_id(1);

  if (row >= numCRows || col >= numCColumns)
    return;

  if (row < numCRows && col < numCColumns) {
    int sum = 0;

    for (unsigned int k = 0; k < numAColumns; k++) {
      sum += A[row * numAColumns + k] * B[k * numBColumns + col];
    }

    C[row * numCColumns + col] = sum;
  }
}
