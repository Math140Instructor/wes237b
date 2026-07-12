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

typedef struct Sphere {
    float radius;
    float3 ambient;

    float3 center;
    float3 diffuse;
    float3 specular;
    float shininess;
} Sphere;

void intersectSphere(Ray* r_ray, Sphere* sphere) {
    float3 p0 = r_ray->origin;
    float3 p1 = r_ray->dir;
    float3 dir_to_center = p0 - sphere->center;

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

    float3 intersectPoint = p0 + p1 * final_t;

    // If a closer intersection is found 
    if (final_t < r_ray-> t) {
        r_ray->t = final_t;
        r_ray->normal = normalize(intersectPoint - sphere->center);

        // keep track of material properties of the new intersection
        r_ray->ambient = sphere->ambient;
        r_ray->diffuse = sphere->diffuse;
        r_ray->specular = sphere->specular;
        r_ray->shininess = sphere->shininess;
    }

}

typedef struct Light {
    float3 pos;
    float3 color;
    float3 atten;
    int dir; // track if point light or not (1 if point, 0 if direc)

    //point lights atten (0,0,1)
    //no atten is (1,0,0)
} Light;

float3 calcLight(float3 surfacePos, Light light) {
    float3 atten = light.atten;
    const float dist = distance(surfacePos, light.pos);
    const float denom = atten.x + atten.y * dist + atten.z * pown(dist, 2);
    //printf("denom: %f\n", denom);

    return light.color / denom;
}

// intesect the ray with every sphere in the scene
void intersectScene(Sphere spheres[], size_t num_spheres, Ray* r_ray) {
    for (int s = 0; s < num_spheres; ++s) {
        intersectSphere(r_ray, &spheres[s]);
    }
}

// reflect the unit vector subject across unit vector mirror
float3 reflect(float3 subject, float3 mirror) {
    return (2.0f * dot(subject, mirror) * mirror) - subject;
}

typedef struct RayHit {
    float3 phong; 
    float3 specular;
} RayHit;
// Shades a ray at its nearest intersection position.
// Assumes the ray has already been intersected with relevant geometry. 
// The ray is passed by value because we'll need to use all its members anyway.
float3 shadeRayHit(Ray ray, Light lights[], size_t num_lights, Sphere spheres[], size_t num_spheres) {
    // return ray.ambient;
    const float3 hit_point = ray.origin + ray.dir * ray.t;
    const float3 hit_normal = ray.normal;

    // total incoming light from all lights
    float3 color = (float3)(0, 0, 0);
    // accumulate brightness from all lights
    for (int l = 0; l < num_lights; ++l) {
        Light curr_light = lights[l];

        // get direction from the hit point to the light
        float3 hit_to_light;
        if (curr_light.dir) {
            hit_to_light = curr_light.pos - hit_point; // point light
        } else {
            hit_to_light = -curr_light.pos; // directional light
        }
        float3 light_direction = normalize(hit_to_light);

        // create a ray from the hit point to the light
        Ray shadow_ray = {
            .origin = hit_point + (hit_normal * 5.0f * 10e-5f), 
            .dir = hit_to_light, 
            .t = INFINITY
        };

        // check to see if the ray can reach the current light source
        intersectScene(spheres, num_spheres, &shadow_ray);
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
        float3 reflection_direction = reflect(light_direction, hit_normal);

        // phong shading model
        float NdotL =  fmax(0, dot(hit_normal, light_direction));
        float3 color_diffuse = ray.diffuse * NdotL;

        float3 color_specular = 
            ray.specular * 
            pow(fmax(0, dot(reflection_direction, -normalize(ray.dir))), ray.shininess);
        color = color + ((color_diffuse + color_specular) * curr_light.color);
    }
    return color + ray.ambient;
}


#define MAX_SPHERES 10 
#define MAX_LIGHTS 5
#define MAX_RECURSION_DEPTH 6

//! KERNEL BEGINNING
__kernel void renderColor(__global unsigned char* output, int img_size, float half_height) {
    int workitem_id = get_global_id(0);

    int row = workitem_id / img_size;
    int col = workitem_id % img_size;

    Sphere spheres[MAX_SPHERES];
    Light lights[MAX_LIGHTS];
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

    // dont forget to set the number of spheres/lights
    unsigned int num_lights = 2; 

     //* ----------------- RAY GENERATION -----------------------------
    float offset_x = half_height * ((col + 0.5f - img_size/2.0f)/(img_size/2.0f));
    float offset_y = half_height * ((img_size/2.0f - row - 0.5f)/(img_size/2.0f));

    float3 ray_direction = (float3)(offset_x, offset_y, -1.0f); 
    ray_direction = normalize(ray_direction);

    //* ----------------- RECURSIVE RAY TRACING -----------------------------
    // set up a stack of ray hit positions along a path
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
        intersectScene(spheres, num_spheres, &curr_ray);

        // ray missed the scene so use the sky shader
        if (isinf(curr_ray.t)) {
            RayHit ray_hit = {
                .phong = {0.5 * curr_ray.dir.x + 0.5, 0.5 * curr_ray.dir.y + 0.5, 0.7 * max(curr_ray.dir.z, -curr_ray.dir.z) + 0.3}, 
                .specular = {0, 0, 0}
            };
            STACK_PUSH(ray_hit)
            break;
        }

        RayHit ray_hit = {
            .phong = shadeRayHit(curr_ray, lights, num_lights, spheres, num_spheres), 
            .specular = curr_ray.specular
        };

        STACK_PUSH(ray_hit);

        float3 reflection_direction = reflect(-normalize(curr_ray.dir), curr_ray.normal);

        // replace the ray for the next loop
        curr_ray = (Ray){
            .origin = (curr_ray.origin + (curr_ray.dir * curr_ray.t)) 
                + (curr_ray.normal * 2.0f * 10e-5f),
            .dir = reflection_direction,
            .t = INFINITY 
        };
    }

    // accumulate color backwards along the path
    float3 final_color = (float3)(0.0f, 0.0f, 0.0f);
    while (ray_hits_top >= 0) {
        // pop from the top of the stack
        RayHit curr_hit = ray_hits[ray_hits_top]; 
        ray_hits_top--;

        final_color *= curr_hit.specular; 
        final_color += curr_hit.phong;
    }

    // map final_color from [0, infinity] -> [0, 255]
    final_color = final_color * 255.0f;
    unsigned char r = fmin(final_color.x, 255.0f); 
    unsigned char g = fmin(final_color.y, 255.0f); 
    unsigned char b = fmin(final_color.z, 255.0f);

    // write to output
    output[(row * img_size + col) * 3 + 0] = r;
    output[(row * img_size + col) * 3 + 1] = g;
    output[(row * img_size + col) * 3 + 2] = b;
}