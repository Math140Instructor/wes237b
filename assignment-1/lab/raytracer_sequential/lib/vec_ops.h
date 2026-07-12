#pragma once

#include <math.h>
#include "float3.h"

// Vector addition
// @param a : first vector operand
// @param b : second vector operand
float3 add(float3 a, float3 b);

// Vector negation
// @param a : vector to negate
float3 neg(float3 a);

// Scalar multiplication
// @param a : vector to scale
// @param scale : scalar
float3 scale(float3 a, float scale);

// Dot product
// @param a : first vector operand
// @param b : second vector operand
float dot(float3 a, float3 b);

// Normalize vector
// @param a : vector to normalize
float3 normalize(float3 a);

// Get distance between two vectors
// @param a : first vector operand
// @param b: second vector operand
float distance(float3 a, float3 b);

// Component wise multiplication of two vectors
// @param a : first vector operand
// @param b: second vector operand
float3 multiply(float3 a, float3 b);

float3 random_param(float min, float max);

float3 rand_on_hemisphere(float3 normal);

float3 clamp(float3 a);

void printfloat3(char* name, float3 vec);