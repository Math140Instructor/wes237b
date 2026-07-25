# Assignment 2 (assigned 7/26, due 8/10)

This assignment will be using the ARM CPU in your RB3.

## Lab

### Loop Demo
This is the starter code for this project.  It loads a vector from file and sums each element into a single result.  It then saves the final result.

### Loop Unrolling Demo
The purpose of this portion of the lab is to look into loop unrolling and how it can impact the performance of the code executed.  This is an optimization that can be perfromed by `gcc` and therefore may not have a visible impact.

### ARM Neon Intrinsics Demo
Using the previous code from the loop unrolling section, we should update this to use the ARM Neon intrinsics.  Please see the resources.

## Homework

### Naive Matrix Multiply
The goal of this portion of the assignment is to provide a baseline, simple implementation of a matrix multiply.

### Block Matrix Multiply
The goal of this portion of the assignment is to improve the performance of this by using block matrix multiplication.

## Unrolled Block Matrix Multiply
The goal of this portion of the assignment is to perform some loop unrolling in preparation for ARM Neon implementation of block matrix multiplication.

## Neon Block Matrix Multiply
The goal of this portion of the assignment is to using ARM Neon instead of loop unrolling for vectorization.

## Resources

* [ARM Neon Instruction Reference](https://developer.arm.com/architectures/instruction-sets/intrinsics/#f:@navigationhierarchiessimdisa=%5BNeon%5D&f:@navigationhierarchiesreturnbasetype=%5Bfloat%5D&f:@navigationhierarchieselementbitsize=%5B32%5D)
