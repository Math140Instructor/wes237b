// high resolution image downscaled and converted to grayscale

__kernel void bgr_to_gray_downscale_2x(
    __global const uchar* restrict src, // input image data
    __global uchar* restrict dst,       // output image in grayscale
    const int src_width,                // image pixel width
    const int src_height,               // image pixel height
    const int src_step,                 // input row stride in bytes 
    const int dst_step                  // output row stride in bytes
) {

    // obtain 2D coordinates of GPU thread
    int out_x = get_global_id(0);
    int out_y = get_global_id(1);

    // calculate dimensions for downsized output image 
    int out_width  = src_width >> 1;
    int out_height = src_height >> 1;

    // prevents threads outside output image bounds from executing 
    if (out_x >= out_width || out_y >= out_height) {
        return;
    }

    int in_x = out_x << 1;
    int in_y = out_y << 1;

    // Byte offsets for the 2x2 pixel quad in BGR format
    // calculate memory byte offsets for each pizel in the 2x2 source
    int idx00 = in_y * src_step + in_x * 3;                 // top-left pixel
    int idx01 = in_y * src_step + (in_x + 1) * 3;           // top-right pixel
    int idx10 = (in_y + 1) * src_step + in_x * 3;           // bottom-left pixel
    int idx11 = (in_y + 1) * src_step + (in_x + 1) * 3;     // bottom-right pixel

    // Convert each of the four source BGR pixels to grayscale using BT.601 math to convert color to grayscale
    // Weights: Red * 77, Green * 150, Blue * 29. Adding 128 handles rounding before shifting right by 8 (/256).
    // https://stackoverflow.com/questions/17615963/standard-rgb-to-grayscale-conversion
    uint y00 = (77u * src[idx00 + 2] + 150u * src[idx00 + 1] + 29u * src[idx00] + 128u) >> 8;
    uint y01 = (77u * src[idx01 + 2] + 150u * src[idx01 + 1] + 29u * src[idx01] + 128u) >> 8;
    uint y10 = (77u * src[idx10 + 2] + 150u * src[idx10 + 1] + 29u * src[idx10] + 128u) >> 8;
    uint y11 = (77u * src[idx11 + 2] + 150u * src[idx11 + 1] + 29u * src[idx11] + 128u) >> 8;

    // Computes average of 4 grayscale values together
    // add 2 handles the integer rounding math before dividing by 4 
    uchar out_val = (uchar)((y00 + y01 + y10 + y11 + 2) >> 2);
    dst[out_y * dst_step + out_x] = out_val;
}


// kernel that draws color-coded rectangular borders directly into the 720p BGR frame in GPU memory
__kernel void draw_box_borders_gpu(
    __global uchar* restrict frame,         // pointer to the frame data 
    const int frame_width,                  // frame pixel width
    const int frame_height,                 // grame pixel height
    const int frame_step,                   // row stries of the frame in bytes
    __global const int* restrict box_data,  // flat array of box data: [bx, by, bw, bh, valid, b, g, r] * num_boxes
    const int num_boxes,                    // total number of boxes in the array
    const int thickness                     // thickness of the box border in pixels
) {

    // obtain 2D coordinates of GPU thread
    int x = get_global_id(0);
    int y = get_global_id(1);

    // prevents threads outside output image bounds from executing 
    if (x >= frame_width || y >= frame_height) {
        return;
      }
    }

    // loops through every bounding box array to see if pixel is in it
    #pragma unroll
    for (int i = 0; i < num_boxes; ++i) {

        // checks if box is not marked valid, skip if so
        if (box_data[i * 8 + 4] == 0) continue;

        // extract spatial dimensions for current bounding box
        int bx = box_data[i * 8 + 0];   // x coordinates
        int by = box_data[i * 8 + 1];   // y coordinates
        int bw = box_data[i * 8 + 2];   // box width
        int bh = box_data[i * 8 + 3];   // box height

        // Check if pixel (x, y) falls inside the outer bounding box
        if (x >= bx && x < bx + bw && y >= by && y < by + bh) {
            // Check if pixel falls on the outer border perimeter defined by thickness
            bool on_left   = (x < bx + thickness);
            bool on_right  = (x >= bx + bw - thickness);
            bool on_top    = (y < by + thickness);
            bool on_bottom = (y >= by + bh - thickness);

            // draws colors if pixel is on any of the borders
            if (on_left || on_right || on_top || on_bottom) {
                // calculate 1D array index for current pixel color data
                int pixel_idx = y * frame_step + x * 3;
                frame[pixel_idx + 0] = (uchar)box_data[i * 8 + 5]; // B
                frame[pixel_idx + 1] = (uchar)box_data[i * 8 + 6]; // G
                frame[pixel_idx + 2] = (uchar)box_data[i * 8 + 7]; // R

                // ends loops if color is drawn
                return;
            }
        }
    }
}