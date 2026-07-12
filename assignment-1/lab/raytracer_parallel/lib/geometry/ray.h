#pragma once
#include <CL/cl.h>



typedef struct Ray {
    cl_float3 origin;
    cl_float3 dir;
    cl_float3 normal;

    // Material properties of closest intersection
    cl_float3 diffuse;
    cl_float3 specular;
    cl_float3 ambient;
    cl_float shininess;
    
    cl_float  t;
} Ray;