#include "Light.h"

// Function to get the light color after accounting for attenuation
float3 calcLight(float3 surfacePos, Light light) {
    float3 atten = light.atten;
    const float dist = distance(surfacePos, light.pos);
    const float denom = atten.x + atten.y * dist + atten.z * powf(dist, 2);
    //printf("denom: %f\n", denom);

    return scale(light.color,  1 / denom);
}