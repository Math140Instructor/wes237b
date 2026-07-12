#include "Sphere.h"

void intersectSphere(Ray* r_ray, Sphere* sphere) {
    float3 p0 = r_ray->origin;
    float3 p1 = r_ray->dir;
    float3 dir_to_center = add(p0, neg(sphere->center));

    // set up quadratic coefficients
    const float a = dot(p1, p1);
    const float b = 2 * dot(p1, dir_to_center);
    const float c = dot(dir_to_center, dir_to_center) - (sphere->radius * sphere->radius);

    const float disc = (b * b) - 4 * a * c;

    // solve ray-sphere system
    const float t1 = (-b + sqrt(disc)) / (2 * a);
    const float t2 = (-b - sqrt(disc)) / (2 * a);

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

    float3 intersectPoint = add(p0, scale(p1, final_t));

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