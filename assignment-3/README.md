# Assignment 3
The focus of this assignment is to take your matrix multiplication from Assignment 2 and adapt it to run on OpenCL.

## Lab

### ARM Neon Intrinsics Demo
Starting from the summing loop in `./lab/neon/main.c`, we should update this to use the ARM Neon intrinsics.  Please see the resources.  For the steps for this lab, see `./lab/neon/README.md`.

## Homework
The goal of this is to accelerate your matrix multiply from Assignment 2.  You may directly use code from the previous assignment here.

### Neon Block Matrix Multiply
The first part of this homework is to implement a block matrix multiply using ARM Neon for vectorization.  This gives you a CPU SIMD baseline to compare the OpenCL versions against.  For the steps for this, see `./homework/block_mm_neon/README.md`.

### OpenCL Naive MM
The second part of this homework assignment is to adapt your naive matrix multiplication to OpenCL.  For the steps for this, see `./homework/opencl_naive_mm/README.md`.

### OpenCL Tiled MM
The third part of this homework is to take your OpenCL naive matrix multiplication and apply OpenCL optimizations.  For the steps for this, see `./homework/opencl_tiled_mm`.

## Resources

* [ARM Neon Instruction Reference](https://developer.arm.com/architectures/instruction-sets/intrinsics/#f:@navigationhierarchiessimdisa=%5BNeon%5D&f:@navigationhierarchiesreturnbasetype=%5Bfloat%5D&f:@navigationhierarchieselementbitsize=%5B32%5D)