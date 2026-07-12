#include "Light.h"

// Function to get the light color after accounting for attenuation
cl_float3 calcLight(cl_float3 surfacePos, Light light) {
    cl_float3 atten = light.atten;
    const cl_float dist = distance(surfacePos, light.pos);
    const cl_float denom = atten.x + atten.y * dist + atten.z * powf(dist, 2);
    //printf("denom: %f\n", denom);

    return scale(light.color,  1 / denom);
}