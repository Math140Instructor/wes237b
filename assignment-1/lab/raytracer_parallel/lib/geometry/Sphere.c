
#include "Sphere.h"



void intersectSphere(Ray* r_ray, Sphere* sphere) {
    cl_float3 p0 = r_ray->origin;
    cl_float3 p1 = r_ray->dir;
    cl_float3 dir_to_center = add(p0, neg(sphere->center));

    // set up quadratic coefficients
    const cl_float a = dot(p1, p1);
    const cl_float b = 2 * dot(p1, dir_to_center);
    const cl_float c = dot(dir_to_center, dir_to_center) - (sphere->radius * sphere->radius);

    const cl_float disc = (b * b) - 4 * a * c;

    // solve ray-sphere system
    const cl_float t1 = (-b + sqrt(disc)) / (2 * a);
    const cl_float t2 = (-b - sqrt(disc)) / (2 * a);

    // Do nothing if no intersection
    if (disc < 0) {
        return;
    }

    float final_t = 0.0f;

    // 1 root
    if (disc == 0) {
        final_t = t1; 
    }
    // lesser root is positive, so use it
    else if (t2 > 0) {
       final_t = t2 ;
    }
    // lesser root is negative and larger root is positive 
    // use the positive root
    else if (t1 > 0) {
        final_t = t1;
    }
    // both roots are negative
    else {
        return;
    }

    cl_float3 intersectPoint = add(p0, scale(p1, final_t));

    // If a closer intersection is found 
    if (final_t < r_ray-> t) {
        r_ray->t = final_t;
        r_ray->normal = normalize(add(intersectPoint, neg(sphere->center)));

        // keep track of material properties of the new intersection
        r_ray->ambient = sphere->ambient;
        r_ray->diffuse = sphere->diffuse;
        r_ray->specular = sphere->specular;
        r_ray->shininess = sphere->shininess;
    }

}