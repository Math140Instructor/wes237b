#include <stdlib.h>
#include "vec_ops.h"


//? May need to define more niche vector operations here

//TODO: Dot product, vector scalar multiplication, vector addition, vector negate, normalize


// Helper functions for random generation
cl_float random_float() {
    return rand() / (RAND_MAX + 1.0f);
}

cl_float random_float_param(cl_float min, cl_float max) {
    return min + (max - min) * random_float();
}

// Vector addition
// @param a : first vector operand
// @param b : second vector operand
cl_float3 add(cl_float3 a, cl_float3 b) {
    cl_float3 sum = {{a.x + b.x, a.y + b.y, a.z + b.z}};
    return sum;
}

// Vector negation
// @param a : vector to negate
cl_float3 neg(cl_float3 a) {
    cl_float3 negated = {{-a.x, -a.y, -a.z}};
    return negated;
}

// Scalar multiplication
// @param a : vector to scale
// @param scale : scalar
cl_float3 scale(cl_float3 a, cl_float scale) {
    cl_float3 scaled = {{scale * a.x, scale * a.y, scale * a.z}};
    return scaled;
}

// Dot product
// @param a : first vector operand
// @param b : second vector operand
cl_float dot(cl_float3 a, cl_float3 b) {
    cl_float dot_prod = (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
    return dot_prod;
}

// Normalize vector
// @param a : vector to normalize
cl_float3 normalize(cl_float3 a) {
    cl_float magnitude = sqrt(dot(a, a));
    cl_float3 normalized_vec = {{a.x / magnitude, a.y / magnitude, a.z / magnitude}};
    return normalized_vec;
}

// Get distance between two vectors
// @param a : first vector operand
// @param b: second vector operand
float distance(cl_float3 a, cl_float3 b) {
    return sqrt(powf(a.x - b.x, 2) + powf(a.y - b.y, 2) + powf(a.z - b.z, 2));
}

// Component wise multiplication of two vectors
// @param a : first vector operand
// @param b: second vector operand
cl_float3 multiply(cl_float3 a, cl_float3 b) {
    cl_float3 multiplied = {{a.x * b.x, a.y * b.y, a.z * b.z}};
    return multiplied;
}


float lengthsquare(cl_float3 a) {
    return pow(a.x, 2) + pow(a.y, 2) + pow(a.z, 2);
}

cl_float3 random_param(float min, float max) {
    return (cl_float3){random_float_param(min, max), random_float_param(min, max), random_float_param(min, max)};
}

cl_float3 rand_in_unit_sphere() {
    while (1) {
        cl_float3 p = random_param(-1, 1);
        if (lengthsquare(p) < 1) {
            return p;
        }
    }
}

cl_float3 rand_unit_vec() {
    return normalize(rand_in_unit_sphere());
}

cl_float3 rand_on_hemisphere(cl_float3 normal) {
    cl_float3 unit_sphere = rand_unit_vec();

    if (dot(unit_sphere, normal) > 0.0f) {
        return unit_sphere;
    } else{
        return neg(unit_sphere);
    }
}

cl_float3 clamp(cl_float3 a) {
    cl_float3 clamped = {{fmin(a.x, 255.0f), fmin(a.y, 255.0f), fmin(a.z, 255.0f)}};

    return clamped;
}

