// kernel.cl - Fused BGR to Grayscale and 2x Box Downsampling Kernel
__kernel void bgr_to_gray_downscale_2x(
    __global const uchar* restrict src,
    __global uchar* restrict dst,
    const int src_width,
    const int src_height,
    const int src_step, // Input row stride in bytes (e.g., 1280 * 3)
    const int dst_step  // Output row stride in bytes (e.g., 640)
) {
    int out_x = get_global_id(0);
    int out_y = get_global_id(1);

    int out_width  = src_width >> 1;
    int out_height = src_height >> 1;

    if (out_x >= out_width || out_y >= out_height) {
        return;
    }

    int in_x = out_x << 1;
    int in_y = out_y << 1;

    // Byte offsets for the 2x2 pixel quad in BGR format
    int idx00 = in_y * src_step + in_x * 3;
    int idx01 = in_y * src_step + (in_x + 1) * 3;
    int idx10 = (in_y + 1) * src_step + in_x * 3;
    int idx11 = (in_y + 1) * src_step + (in_x + 1) * 3;

    // Convert BGR to Grayscale (BT.601 integer fixed-point): Y = (77*R + 150*G + 29*B + 128) >> 8
    // Memory layout: [0]=Blue, [1]=Green, [2]=Red
    uint y00 = (77u * src[idx00 + 2] + 150u * src[idx00 + 1] + 29u * src[idx00] + 128u) >> 8;
    uint y01 = (77u * src[idx01 + 2] + 150u * src[idx01 + 1] + 29u * src[idx01] + 128u) >> 8;
    uint y10 = (77u * src[idx10 + 2] + 150u * src[idx10 + 1] + 29u * src[idx10] + 128u) >> 8;
    uint y11 = (77u * src[idx11 + 2] + 150u * src[idx11 + 1] + 29u * src[idx11] + 128u) >> 8;

    // Compute 2x2 box average
    uchar out_val = (uchar)((y00 + y01 + y10 + y11 + 2) >> 2);

    dst[out_y * dst_step + out_x] = out_val;
}