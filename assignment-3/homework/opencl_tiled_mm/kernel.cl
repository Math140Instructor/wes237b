#define TILE_SIZE 16

__kernel void matrixMultiply(__global const int *A, __global const int *B, __global int *C, const unsigned int numARows, const unsigned int numAColumns, const unsigned int numBRows, const unsigned int numBColumns, const unsigned int numCRows, const unsigned int numCColumns) {
  const size_t col = get_global_id(0);
  const size_t row = get_global_id(1);

  const size_t localCol = get_local_id(0);
  const size_t localRow = get_local_id(1);

  __local int tileA[TILE_SIZE][TILE_SIZE];
  __local int tileB[TILE_SIZE][TILE_SIZE];

  int sum = 0;

  const unsigned int numberOfTiles = (numAColumns + TILE_SIZE - 1) / TILE_SIZE;

  for (unsigned int tile = 0; tile < numberOfTiles; ++tile) {
    const size_t aCol = tile * TILE_SIZE + localCol;
    const size_t bRow = tile * TILE_SIZE + localRow;

    // Load one A value into local memory
    if (row < numARows && aCol < numAColumns)
      tileA[localRow][localCol] = A[row * numAColumns + aCol];
    else
      tileA[localRow][localCol] = 0;

    // Load one B value into local memory
    if (bRow < numBRows && col < numBColumns)
      tileB[localRow][localCol] = B[bRow * numBColumns + col];
    else
      tileB[localRow][localCol] = 0;

    // Wait until the entire tile has been loaded
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int k = 0; k < TILE_SIZE; ++k)
      sum += tileA[localRow][k] * tileB[k][localCol];

    // Finish using the current tile before overwriting it
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (row < numCRows && col < numCColumns)
    C[row * numCColumns + col] = sum;
}