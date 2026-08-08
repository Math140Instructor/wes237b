#pragma once

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#define CL_TARGET_OPENCL_VERSION 300 // Use OpenCL 3.0
#include <CL/cl.h>
#endif

#define IMAGE_CHANNELS 3

typedef struct _Image 
{
    int *data;
    unsigned int shape[IMAGE_CHANNELS];
} Image;

cl_int LoadImg(const char *path, Image* img);
cl_int LoadStride(const char *dir, int *stride);
cl_int LoadImgRaw(const char *path, Image* img);
cl_int SaveImg(const char *path, Image *matrix);
cl_int CheckImg(Image *truth, Image *student);
