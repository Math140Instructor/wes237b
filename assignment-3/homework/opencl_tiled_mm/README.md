# Tiled Matrix Multiplication
## Objective
Implement a tiled dense matrix multiplication routine using shared memory.

## Instructions
Edit the code in the code tab to perform the following:

* Allocate device memory
* Copy host memory to device
* Initialize work group dimensions
* Invoke OpenCL kernel
* Copy results from device to host
* Deallocate device memory
* Implement the matrix-matrix multiplication routine using shared memory and tiling

Recommended Block dimensions are $16 \times 16$.

## Files and Directories

* `Makefile`: Automates the compilation and execution process for the project.
* `main.c`: Source code for the tiled matrix multiplication using OpenCL.
* `kernel.cl`: OpenCL kernel file for performing tiled matrix multiplication.

## How to Compile
The `main.c`, `kernel.cl` files contains the code for the programming assignment. There is a Makefile included which compiles it and links it with the `helper_lib` library automatically. It can be run by typing make from the `opencl_tiled_mm` folder. It generates a `solution` output file.

## How to Test
Use the make run command to test your program. here are a total of 9 tests on which your program will be evaluated for (functional) correctness. Since parts 1, 2, and 3 are functionally the same, we will use the last test case to verify if your programs meet the speedup requirements that you should get using shared memory.

## Submission
Submit the GitHub repo to Gradescope.

## Further Documentation
`float allocateMatrix(int rows, int cols)`:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Allocates memory for a 2D matrix with the given number of rows and columns. Returns a pointer to the allocated matrix.

`void generateRandomMatrix(int matrix, int rows, int cols)`:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Fills a given matrix with random float values. Each element in the matrix is assigned a random int value.

`int multiplyMatrices(int** A, int A_rows, int A_cols, int** B, int B_cols)`:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Multiplies two matrices $A$ and $B$. $A$ has dimensions $A_{rows} \times A_{cols}$ and $B$ has dimensions $A_{cols} \times B_{cols}$. Returns a new matrix $C$ that is the product of $A$ and $B$ with dimensions $A_{rows} \times B_{cols}$.

`void saveMatrixToFile(const char* filename, float matrix, int rows, int cols)`:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Saves a given matrix to a file specified by filename. The matrix is written to the file with each element separated by a space and each row on a new line.

`**int main(int argc, char* argv[])`:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The entry point of the program. Expects three command-line arguments representing the dimensions of the matrices: $A$ (rows of matrix1), $B$ (columns of matrix1 and rows of matrix2), and $C$ (columns of matrix2). Allocates and generates two random matrices of sizes $A \times B$ and $B \times C$. Multiplies these matrices and saves the result in three files: `A.txt`, `B.txt`, and `C.txt`. Frees the allocated memory for all matrices before exiting.

