#pragma once

#include "ray.h"
#include "../vec_ops.h"
#include <CL/cl.h>


typedef struct Sphere {
    cl_float radius;
    cl_float shininess;
    // 2 dummy variables padded for bit alignment
    cl_float dummy1; 
    cl_float dummy2;
    cl_float3 ambient;
    cl_float3 center;
    cl_float3 diffuse;
    cl_float3 specular;
} Sphere;


void intersectSphere(Ray* r_ray, Sphere* sphere);