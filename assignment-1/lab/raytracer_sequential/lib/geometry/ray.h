#pragma once

#include "../float3.h"

typedef struct Ray {
    float3 origin;
    float3 dir;
    float3 normal;

    // Material properties of closest intersection
    float3 diffuse;
    float3 specular;
    float3 ambient;
    float shininess;
    
    float  t;
} Ray;