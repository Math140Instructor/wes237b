#include <assert.h>
#include <stdio.h>
#include <ctype.h>
#include <stdbool.h>
#include <string.h>

#include "device.h"

#define BYTES_IN_KB 1024
#define BYTES_IN_MB (1024 * 1024)

OclPlatformProp *platforms = NULL;
cl_uint num_platforms;
cl_int status;

int main(void)
{
    // Print this out so that we can tell if this software ran.
    printf("Finding OpenCL Platforms...\n");

    // Find all OpenCL platforms and devices on host machine
    status = OclFindPlatforms((const OclPlatformProp **)&platforms, &num_platforms);
    if (status != CL_SUCCESS)
    {
        printf("Error getting OpenCL platforms\n");
        return -1;
    }

    printf("Found %d platforms!\n", num_platforms);

    for (int i = 0; i < num_platforms; i++)
    {
        printf("Platform %d\n", i);
        printf("- Name: %s\n", platforms[i].name);
        printf("- Version: %s\n", platforms[i].version);
        printf("- Vendor: %s\n", platforms[i].vendor);
        printf("- Num Devices: %u\n", platforms[i].num_devices);

        for (int j = 0; j < platforms[i].num_devices; j++)
        {
            OclDeviceProp device = platforms[i].devices[j];
            printf("\tDevice: %d\n", j);
            printf("\t- Name: %s\n", device.name);
            printf("\t- Type: %s\n", OclDeviceTypeString(*device.type));
            printf("\t- Compute units: %u\n", *device.max_compute_units);
            printf("\t- Global memory size: %lu bytes (%lu MB)\n",
                   *device.global_mem_size, *device.global_mem_size / BYTES_IN_MB);
            printf("\t- Max constant buffer size: %lu bytes (%lu MB)\n",
                   *device.max_constant_buffer_size, *device.max_constant_buffer_size / BYTES_IN_MB);
            printf("\t- Max local memory size per Compute Unit: %lu bytes (%lu KB)\n",
                   *device.local_mem_size, *device.local_mem_size / BYTES_IN_KB);
            printf("\t- Max Work Item Dimensions: %u\n", *device.max_work_item_dimensions);
            printf("\t- Max Work Item size: ");
            for (int d = 0; d < *device.max_work_item_dimensions; d++)
            {
                if (d == 0)
                {
                    printf("%lu ", device.max_work_item_sizes[d]);
                }
                else
                {
                    printf("x %lu ", device.max_work_item_sizes[d]);
                }
            }
            printf("\n");
            printf("\t- Max group size: %lu\n", *device.max_work_group_size);
            printf("\n");
        }
        OclFreePlatformProp(&platforms[i]); // Free memory after printing
    }
    return 0;
}