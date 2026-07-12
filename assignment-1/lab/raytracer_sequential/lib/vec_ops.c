#include <stdlib.h>
#include "vec_ops.h"
#include <stdio.h>

//? May need to define more niche vector operations here

//TODO: Dot product, vector scalar multiplication, vector addition, vector negate, normalize


// Helper functions for random generation
float random_float() {
    return rand() / (RAND_MAX + 1.0f);
}

float random_float_param(float min, float max) {
    return min + (max - min) * random_float();
}

// Vector addition
// @param a : first vector operand
// @param b : second vector operand
float3 add(float3 a, float3 b) {
    float3 sum = {a.x + b.x, a.y + b.y, a.z + b.z};
    return sum;
}

// Vector negation
// @param a : vector to negate
float3 neg(float3 a) {
    float3 negated = {-a.x, -a.y, -a.z};
    return negated;
}

// Scalar multiplication
// @param a : vector to scale
// @param scale : scalar
float3 scale(float3 a, float scale) {
    float3 scaled = {scale * a.x, scale * a.y, scale * a.z};
    return scaled;
}

// Dot product
// @param a : first vector operand
// @param b : second vector operand
float dot(float3 a, float3 b) {
    float dot_prod = (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
    return dot_prod;
}

// Normalize vector
// @param a : vector to normalize
float3 normalize(float3 a) {
    float magnitude = sqrt(dot(a, a));
    float3 normalized_vec = {a.x / magnitude, a.y / magnitude, a.z / magnitude};
    return normalized_vec;
}

// Get distance between two vectors
// @param a : first vector operand
// @param b: second vector operand
float distance(float3 a, float3 b) {
    return sqrt(powf(a.x - b.x, 2) + powf(a.y - b.y, 2) + powf(a.z - b.z, 2));
}

// Component wise multiplication of two vectors
// @param a : first vector operand
// @param b: second vector operand
float3 multiply(float3 a, float3 b) {
    float3 multiplied = {a.x * b.x, a.y * b.y, a.z * b.z};
    return multiplied;
}

float lengthsquare(float3 a) {
    return pow(a.x, 2) + pow(a.y, 2) + pow(a.z, 2);
}

float3 random_param(float min, float max) {
    return (float3){random_float_param(min, max), random_float_param(min, max), random_float_param(min, max)};
}

float3 rand_in_unit_sphere() {
    while (1) {
        float3 p = random_param(-1, 1);
        if (lengthsquare(p) < 1) {
            return p;
        }
    }
}

float3 rand_unit_vec() {
    return normalize(rand_in_unit_sphere());
}

float3 rand_on_hemisphere(float3 normal) {
    float3 unit_sphere = rand_unit_vec();

    if (dot(unit_sphere, normal) > 0.0f) {
        return unit_sphere;
    } else{
        return neg(unit_sphere);
    }
}

float3 clamp(float3 a) {
    float3 clamped = {fmin(a.x, 255.0f), fmin(a.y, 255.0f), fmin(a.z, 255.0f)};

    return clamped;
}

void printfloat3(char* name, float3 vec) {
    printf("%s: %f, %f, %f\n", name, vec.x, vec.y, vec.z);
}