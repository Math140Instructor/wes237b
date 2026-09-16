inline uchar bgr_to_gray(__global const uchar *src, int index) {
  float B = src[index];
  float G = src[index + 1];
  float R = src[index + 2];
  // https://stackoverflow.com/questions/17615963/standard-rgb-to-grayscale-conversion
  return (uchar)(0.114f * B + 0.587f * G + 0.299f * R);
}

// high resolution image downscaled and converted to grayscale
__kernel void bgr_to_gray_downscale_2x(__global const uchar *src, __global uchar *dst, const int src_width, const int src_height, const int src_step, const int dst_step) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  int out_width = src_width / 2;
  int out_height = src_height / 2;

  if (x >= out_width || y >= out_height)
    return;

  // Location of the corresponding 2x2 pixels in the original image
  int src_x = x * 2;
  int src_y = y * 2;

  int top_left = src_y * src_step + src_x * 3;
  int top_right = src_y * src_step + (src_x + 1) * 3;
  int bottom_left = (src_y + 1) * src_step + src_x * 3;
  int bottom_right = (src_y + 1) * src_step + (src_x + 1) * 3;

  // Convert the four pixels to grayscale
  uint gray1 = bgr_to_gray(src, top_left);
  uint gray2 = bgr_to_gray(src, top_right);
  uint gray3 = bgr_to_gray(src, bottom_left);
  uint gray4 = bgr_to_gray(src, bottom_right);

  // Average to produce one pixel in the downsampled 640x360 image
  dst[y * dst_step + x] = (uchar)((gray1 + gray2 + gray3 + gray4) / 4);
}

// kernel that draws color coded rectangular borders directly into the downsampled BGR frame
__kernel void draw_box_borders_gpu(__global uchar *restrict frame,        // pointer to the frame data
                                   const int frame_width,                 // frame pixel width
                                   const int frame_height,                // grame pixel height
                                   const int frame_step,                  // row stries of the frame in bytes
                                   __global const int *restrict box_data, // flat array of box data: [bx, by, bw, bh, valid, b, g, r] * num_boxes
                                   const int num_boxes,                   // total number of boxes in the array
                                   const int thickness                    // thickness of the box border in pixels
) {

  // obtain 2D coordinates of GPU thread
  int x = get_global_id(0);
  int y = get_global_id(1);

  // ensure pixels are inside downsampled bounds
  if (x >= frame_width || y >= frame_height) {
    return;
  }

  // loops through every bounding box array to see if pixel is in it
  // Each box looks like [x, y, width, height, valid, blue, green, red]
  for (int i = 0; i < num_boxes; ++i) {

    // skip if blox is not valid
    if (box_data[i * 8 + 4] == 0)
      continue;

    // extract spatial dimensions for current bounding box
    int bx = box_data[i * 8 + 0]; // x coordinates
    int by = box_data[i * 8 + 1]; // y coordinates
    int bw = box_data[i * 8 + 2]; // box width
    int bh = box_data[i * 8 + 3]; // box height

    // Check if pixel (x, y) falls inside the outer bounding box
    if (x >= bx && x < bx + bw && y >= by && y < by + bh) {
      // Check if pixel falls on the outer border perimeter defined by thickness
      bool on_left = (x < bx + thickness);
      bool on_right = (x >= bx + bw - thickness);
      bool on_top = (y < by + thickness);
      bool on_bottom = (y >= by + bh - thickness);

      // draws colors if pixel is on any of the borders
      if (on_left || on_right || on_top || on_bottom) {
        // calculate 1D array index for current pixel color data
        int pixel_idx = y * frame_step + x * 3;
        frame[pixel_idx + 0] = (uchar)box_data[i * 8 + 5]; // B
        frame[pixel_idx + 1] = (uchar)box_data[i * 8 + 6]; // G
        frame[pixel_idx + 2] = (uchar)box_data[i * 8 + 7]; // R
        return;
      }
    }
  }
}