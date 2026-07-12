#pragma once

#include <math.h>

#include "ray.h"
#include "../vec_ops.h"
#include "../float3.h"

typedef struct Sphere {
    float radius;
    float3 ambient;
    //TODO: pad with two more floats for byte alignment

    float3 center;
    float3 diffuse;
    float3 specular;
    float shininess;
} Sphere;


void intersectSphere(Ray* r_ray, Sphere* sphere);