#pragma once

#include <math.h>
#include <CL/cl.h>

// Vector addition
// @param a : first vector operand
// @param b : second vector operand
cl_float3 add(cl_float3 a, cl_float3 b);

// Vector negation
// @param a : vector to negate
cl_float3 neg(cl_float3 a);

// Scalar multiplication
// @param a : vector to scale
// @param scale : scalar
cl_float3 scale(cl_float3 a, cl_float scale);

// Dot product
// @param a : first vector operand
// @param b : second vector operand
cl_float dot(cl_float3 a, cl_float3 b);

// Normalize vector
// @param a : vector to normalize
cl_float3 normalize(cl_float3 a);

// Get distance between two vectors
// @param a : first vector operand
// @param b: second vector operand
float distance(cl_float3 a, cl_float3 b);

// Component wise multiplication of two vectors
// @param a : first vector operand
// @param b: second vector operand
cl_float3 multiply(cl_float3 a, cl_float3 b);

cl_float3 rng();

cl_float3 random_param(cl_float min, cl_float max);

cl_float3 rand_on_hemisphere(cl_float3 normal);

cl_float3 clamp(cl_float3 a);
