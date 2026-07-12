#include <stdio.h>
#include <math.h>
#include <time.h>
#include "lib/geometry/ray.h"
#include "lib/geometry/Sphere.h"
#include "lib/geometry/Light.h"
#include "lib/vec_ops.h"
#include "lib/float3.h"

#define MAX_SPHERES 10 
#define MAX_LIGHTS 5
#define MAX_RECURSION_DEPTH 5

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"

const float PI = 3.14159265359;
const unsigned int IMG_SIZE = 1024;
const float FOV = 3.14159265359/3.0;

Sphere spheres[MAX_SPHERES];
Light lights[MAX_LIGHTS];

typedef unsigned char bool;
enum {false, true};

void intersectScene(size_t num_spheres, Ray* r_ray) {
    for (int s = 0; s < num_spheres; ++s) {
        intersectSphere(r_ray, &spheres[s]);
    }
}

typedef struct RayHit {
    float3 phong; 
    float3 specular;
} RayHit;
// Shades a ray at its nearest intersection position.
// Assumes the ray has already been intersected with relevant geometry. 
// The ray is passed by value because we'll need to use all its members anyway.
float3 shadeRayHit(Ray ray, size_t num_lights, size_t num_spheres) {
    const float3 hit_point = add(ray.origin, scale(ray.dir, ray.t));
    const float3 hit_normal = ray.normal;

    // total incoming light from all lights
    float3 color = {.x = 0, .y = 0, .z = 0};
    // accumulate brightness from all lights
    for (int l = 0; l < num_lights; ++l) {
        Light curr_light = lights[l];

        // get direction from the hit point to the light
        float3 hit_to_light;
        if (curr_light.dir) {
            hit_to_light = add(curr_light.pos, neg(hit_point)); // point light
        } else {
            hit_to_light = neg(curr_light.pos); // directional light
        }
        float3 light_direction = normalize(hit_to_light);

        // create a ray from the hit point to the light
        Ray shadow_ray = {
            .origin = add(hit_point, scale(hit_normal, 5 * 10e-5)), 
            .dir = hit_to_light, 
            .t = INFINITY
        };

        // check to see if the ray can reach the current light source
        intersectScene(num_spheres, &shadow_ray);
        bool light_reached;
        if(curr_light.dir) {
            // point light 
            light_reached = shadow_ray.t > 1;
        } else {
            // directional light
            light_reached = isinf(shadow_ray.t);
        }
        
        // something blocked the shadow ray from reaching the light
        if (!light_reached) {
            continue;
        }

        // the light is reachable so we evaluate the shading model 
        float3 reflection_direction = add(
            scale(
                hit_normal, 
                dot(light_direction, hit_normal) * 2.0f), 
            neg(light_direction)
        );

        // phong shading model
        float NdotL =  fmax(0, dot(hit_normal, light_direction));
        float3 color_diffuse = scale(ray.diffuse, NdotL);
        float3 color_specular = scale(
            ray.specular, 
            pow(
                fmax(0, dot(reflection_direction, neg(normalize(ray.dir)))), 
                ray.shininess)
        );

        color = add(
            color, 
            multiply(
                add(color_diffuse, color_specular), 
                curr_light.color)
        );
    }
    return add(color, ray.ambient);
}

;int main (int argc, char *argv[]) {
    printf("Starting Sequential Ray Tracing...");

    // Time measurement variables
    clock_t start, end;
    double cpu_time_used;

    // Start measuring host execution time
    start = clock();

       // hardcode the scene for now
    spheres[0] = (Sphere){
        .radius = 1.3f, 
        .center = (float3){3.4, -1.8, -9},
        .ambient = (float3){0.2, 0.1, 0.2},
        .diffuse = (float3){.7, 0.0, 0.0}, 
        
        .specular = (float3){0.3, 0.1, 0.1}, 
        .shininess = 100 
    };

    spheres[1] = (Sphere){
        .radius = 3.0f, 
        .center = (float3){2, 0, -18},
        .ambient = (float3){0.2, 0.1, 0.2},
        .diffuse = (float3){0.7, 0.6, 0.9}, 
        .specular = (float3){0.6, 0.6, 0.6}, 
        .shininess =  100 
    };

    // Ground sphere
    spheres[2] = (Sphere){
        .radius = 2500.0f, 
        .center = (float3){0, -2501.5, -100},
        .ambient = (float3){0.1, 0.05, 0.1},
        .diffuse = (float3){0.5, 0.5, 0.5},     
        .specular = (float3){0.4, 0.4, 0.4},    
        .shininess = 20
    };

    spheres[3] = (Sphere){
        .radius = 1.3f, 
        .center = (float3){-3, -2, -7},
        .ambient = (float3){0.2, 0.1, 0.2},
        .diffuse = (float3){0.0, 0.7, 0.7}, 
        .specular = (float3){0.1, 0.5, 0.1}, 
        .shininess = 10 
    };
    spheres[4] = (Sphere){
        .radius = 0.5f, 
        .center = (float3){-1.8, -2.5, -10},
        .ambient = (float3){0.1, 0.2, 0.2},
        .diffuse = (float3){0.0, 0.6, 0.5},
        .specular = (float3){0.1, 0.1, 0.1}, 
        .shininess = 100 
    };
    spheres[5] = (Sphere){
        .radius = 0.6667f, 
        .center = (float3){-1.6, -2.7, -6},
        .ambient = (float3){0, 0, 0.5},
        .diffuse = (float3){0.0, 0.7, 0.7},
        .specular = (float3){0.1, 0.1, 0.2}, 
        .shininess = 0 
    };
    spheres[6] = (Sphere){
        .radius = 0.4f, 
        .center = (float3){1.5, -2.8, -7.2},
        .ambient = (float3){0.4, 0, 0.0},
        .diffuse = (float3){0.7, 0.7, 0.0},
        .specular = (float3){0.1, 0.1, 0.2}, 
        .shininess = 0 
    };
    spheres[7] = (Sphere){
        .radius = 0.5f, 
        .center = (float3){2.4, -2.7, -6.5},
        .ambient = (float3){0.2, 0.0, 0.2},
        .diffuse = (float3){0.7, 0.0, 0.4},
        .specular = (float3){0.1, 0.1, 0.1}, 
        .shininess = 50 
    };
    spheres[8] = (Sphere){
        .radius = 5.0f, 
        .center = (float3){-6, 2, -15},
        .ambient = (float3){0.2, 0.1, 0.2},
        .diffuse = (float3){0.7, 0.7, 0.7},
        .specular = (float3){0.4, 0.4, 0.4}, 
        .shininess = 100 
    };
    
    // don't forget to update this
    unsigned int num_spheres = 9;

    // Lights
    lights[0] = (Light){
        .pos = (float3){8, 4, 7},
        .color = (float3){0.5, 0.5, 0.5},
        .atten = (float3){1, 0, 0},  
        .dir = 1
    };

    lights[1] = (Light){
        .pos = (float3){-5, 5, 0},
        .color = (float3){0.1, 0.1, 0.1},
        .atten = (float3){1, 0, 0},
        .dir = 1
    };
    // Set the number of spheres/lights
    unsigned int num_lights = 2; 
    
    // Pixels to write to the image
    unsigned char pixels[IMG_SIZE * IMG_SIZE * 3];

    // half the height (and also width) of the image plane
    float half_height = tan(FOV * 0.5);

    // Iterate through every pixel on the screen; for every row, iterate through
    // every column.  All pixels are drawn on one thread. 
    for (int row = 0; row < IMG_SIZE; row++) {
        for (int col = 0; col < IMG_SIZE; col++) {

            //* ----------------- RAY GENERATION  -----------------------------
            float offset_x = half_height * ((col + 0.5 - IMG_SIZE/2.0)/(IMG_SIZE/2.0));
            float offset_y = half_height * ((IMG_SIZE/2.0 - row - 0.5)/(IMG_SIZE/2.0));

            float3 ray_direction = {.x = offset_x, .y = offset_y, .z = -1}; 
            ray_direction = normalize(ray_direction);

            //* ----------------- RECURSIVE RAY TRACING -----------------------------
            RayHit ray_hits[MAX_RECURSION_DEPTH];
            int ray_hits_top = -1; 
            #define STACK_PUSH(value) ray_hits_top++; ray_hits[ray_hits_top] = value;

            // changes at each iteration of the loop
            Ray curr_ray = {
                .origin = (float3){0.0, 0.0, 0.0}, 
                .dir = ray_direction, 
                .t = INFINITY 
            };

            for (int i = 0; i < MAX_RECURSION_DEPTH; ++i) {
                intersectScene(num_spheres, &curr_ray);

                // ray missed the scene so use the sky shader
                if (isinf(curr_ray.t)) {
                    RayHit ray_hit = {
                        .phong = {0.5 * curr_ray.dir.x + 0.5, 0.5 * curr_ray.dir.y + 0.5, 0.7 * fabsf(curr_ray.dir.z) + 0.3}, 
                        .specular = {0, 0, 0}
                    };
                    STACK_PUSH(ray_hit)
                    break;
                }

                // Intersection for shading
                RayHit ray_hit = {
                    .phong = shadeRayHit(curr_ray, num_lights, num_spheres), 
                    .specular = curr_ray.specular
                };

                STACK_PUSH(ray_hit);

                // reflect the unit vector from the surface to the ray origin across the normal
                float3 reflection_direction = add(
                    scale(
                        curr_ray.normal, 
                        dot(neg(normalize(curr_ray.dir)), curr_ray.normal) * 2.0f), 
                    neg(neg(normalize(curr_ray.dir)))
                );

                // replace the ray for the next loop
                curr_ray = (Ray){
                    .origin = add(add(curr_ray.origin, scale(curr_ray.dir, curr_ray.t)), scale(curr_ray.normal, 2 * 10e-5)),
                    .dir = reflection_direction,
                   .t = INFINITY 
                };
            }


            float3 final_color = {0.0f, 0.0f, 0.0f};
            // accumulate color backwards along the path
            while (ray_hits_top >= 0) {
                RayHit curr_hit = ray_hits[ray_hits_top]; 
                ray_hits_top--;

                final_color = multiply(final_color, curr_hit.specular); 
                final_color = add(final_color, curr_hit.phong);
            }

            //* ------------------------------- WRITE TO IMAGE --------------------------------------------
            // map final color from [0, infinity) to [0, 255]
            // final_color = (float3){(float)row/IMG_SIZE, (float)col/IMG_SIZE, 0};
            final_color = scale(final_color, 255.0f);
            unsigned char r = fmin(final_color.x, 255.0f); 
            unsigned char g = fmin(final_color.y, 255.0f); 
            unsigned char b = fmin(final_color.z, 255.0f);

            // write the pixels 
            pixels[(row * IMG_SIZE + col) * 3 + 0] = r;
            pixels[(row * IMG_SIZE + col) * 3 + 1] = g;
            pixels[(row * IMG_SIZE + col) * 3 + 2] = b;

        }
    }
    stbi_write_png("output.png", IMG_SIZE, IMG_SIZE, 3, pixels, IMG_SIZE * 3);

    // Stop measuring host execution time
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC * 1000; // Convert to milliseconds

    printf("\n========================================================\n");
    printf("Single-Threaded Implementation on CPU\n");
    printf("Total execution time (host + kernel): %.3f ms\n", cpu_time_used);
     printf("========================================================\n\n");
}