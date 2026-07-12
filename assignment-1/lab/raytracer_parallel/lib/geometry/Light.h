#pragma once

#include <math.h>
#include <CL/cl.h>
#include "../float3.h"
#include "../vec_ops.h"

typedef struct Light {
    cl_float3 pos;
    cl_float3 color;
    cl_float3 atten;
    cl_int dir; // track if point light or not (1 if point, 0 if direc)

    //point lights atten (0,0,1)
    //no atten is (1,0,0)
} Light;

cl_float3 calcLight(cl_float3 surfacePos, Light light);