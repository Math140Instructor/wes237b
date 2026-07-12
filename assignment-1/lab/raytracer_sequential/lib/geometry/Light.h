#pragma once

#include <math.h>
#include "../float3.h"
#include "../vec_ops.h"

typedef struct Light {
    float3 pos;
    float3 color;
    float3 atten;
    int dir; // track if point light or not (1 if point, 0 if direc)

    //point lights atten (0,0,1)
    //no atten is (1,0,0)
} Light;

float3 calcLight(float3 surfacePos, Light light);