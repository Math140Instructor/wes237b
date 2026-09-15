// ============================================================================
// kernel.cl - Fused Preprocessing and GPU Bounding Box Rendering Kernels
// ============================================================================

// ----------------------------------------------------------------------------
// Kernel 1: Fused BGR to Grayscale and 2x Downscale (Preprocessing)
// ----------------------------------------------------------------------------
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

    if50_width || out_y >= out_height) {
      50
    }

    in50 << 1;
    in50 << 1;

    // Byte offsets for the 2x2 pixel quad in BGR format
    int idx00 = in_y * src_step + in_x * 3;
    int idx01 = in_y * src_step + (in_x + 1) * 3;
    int idx10 = (in_y + 1) * src_step + in_x * 3;
    int idx11 = (in_y + 1) * src_step + (in_x + 1) * 3;

    // BT.601 integer fixed-point grayscale: Y = (77*R + 150*G + 29*B + 128) >> 8
    uint y00 = (77u * src[idx00 + 2] + 150u * src[idx00 + 1] + 29u * src[idx00] + 128u) >> 8;
    uint y01 = (77u * src[idx01 + 2] + 150u * src[idx01 + 1] + 29u * src[idx01] + 128u) >> 8;
    uint y10 = (77u * src[idx10 + 2] + 150u * src[idx10 + 1] + 29u * src[idx10] + 128u) >> 8;
    uint y11 = (77u * src[idx11 + 2] + 150u * src[idx11 + 1] + 29u * src[idx11] + 128u) >> 8;

    // Compute 2x2 box average
    uchar out_val = (uchar)((y00 + y01 + y10 + y11 + 2) >> 2);
    dst[out_y * dst_step + out_x] = out_val;
}

// ----------------------------------------------------------------------------
// Kernel 2: GPU Bounding Box Rendering Kernel
// Draws color-coded rectangular borders directly into the 720p BGR frame in GPU memory
// ----------------------------------------------------------------------------
__kernel void draw_box_borders_gpu(
    __global uchar* restrict frame,
    const int frame_width,
    const int frame_height,
    const int frame_step,
    __global const int* restrict box_data, // [bx, by, bw, bh, valid, b, g, r] * num_boxes
    const int num_boxes,
    const int thickness
) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= frame_width || y >= frame_height) {
        return;
    }

    #pragma unroll
    for (int i = 0; i < num_boxes; ++i) {
        // If box is not marked valid, skip
        if (box_data[i * 8 + 4] == 0) continue;

        int bx = box_data[i * 8 + 0];
        int by = box_data[i * 8 + 1];
        int bw = box_data[i * 8 + 2];
        int bh = box_data[i * 8 + 3];

        // Check if pixel (x, y) falls inside the outer bounding box
        if (x >= bx && x < bx + bw && y >= by && y < by + bh) {
            // Check if pixel falls on the outer border perimeter defined by thickness
            bool on_left   = (x < bx + thickness);
            bool on_right  = (x >= bx + bw - thickness);
            bool on_top    = (y < by + thickness);
            bool on_bottom = (y >= by + bh - thickness);

            if (on_left || on_right || on_top || on_bottom) {
                int pixel_idx = y * frame_step + x * 3;
                frame[pixel_idx + 0] = (uchar)box_data[i * 8 + 5]; // B
                frame[pixel_idx + 1] = (uchar)box_data[i * 8 + 6]; // G
                frame[pixel_idx + 2] = (uchar)box_data[i * 8 + 7]; // R
                return;
            }
        }
    }
}