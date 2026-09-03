__kernel void convolution2D(__global int *inputData, __global int *outputData, __constant int *maskData, int width, int height, int maskWidth, int imageChannels, int stride) {
  int outX = get_global_id(0);
  int outY = get_global_id(1);
  int channel = get_global_id(2);

  int localX = get_local_id(0);
  int localY = get_local_id(1);

  int localWidth = get_local_size(0);
  int localHeight = get_local_size(1);

  int outputWidth = (width - maskWidth) / stride + 1;
  int outputHeight = (height - maskWidth) / stride + 1;

  int groupOutX = outX - localX;
  int groupOutY = outY - localY;

  int groupInputX = groupOutX * stride;
  int groupInputY = groupOutY * stride;

  int tileWidth = (localWidth - 1) * stride + maskWidth;

  int tileHeight = (localHeight - 1) * stride + maskWidth;

  int tileSize = tileWidth * tileHeight;

  __local int localData[1024];

  if (tileSize > 1024) {
    if (outX >= outputWidth || outY >= outputHeight || channel >= imageChannels) {
      return;
    }

    int inputX = outX * stride;
    int inputY = outY * stride;

    int accum = 0;

    for (int y = 0; y < maskWidth; y++) {
      for (int x = 0; x < maskWidth; x++) {
        int imageIndex = ((inputY + y) * width + (inputX + x)) * imageChannels + channel;

        int maskIndex = y * maskWidth + x;

        accum += inputData[imageIndex] * maskData[maskIndex];
      }
    }

    int outputIndex = (outY * outputWidth + outX) * imageChannels + channel;

    outputData[outputIndex] = accum;

    return;
  }

  int localIndex = localY * localWidth + localX;

  int localSize = localWidth * localHeight;

  /*
   * ------------------------------------------------
   * GLOBAL -> LOCAL
   *
   * Every work-item helps load the input tile.
   * ------------------------------------------------
   */
  for (int i = localIndex; i < tileSize; i += localSize) {
    int tileX = i % tileWidth;
    int tileY = i / tileWidth;

    int imageX = groupInputX + tileX;
    int imageY = groupInputY + tileY;

    if (imageX < width && imageY < height && channel < imageChannels) {
      int imageIndex = (imageY * width + imageX) * imageChannels + channel;

      localData[i] = inputData[imageIndex];
    } else {
      localData[i] = 0;
    }
  }

  /*
   * All threads must finish loading local memory.
   */
  barrier(CLK_LOCAL_MEM_FENCE);

  /*
   * Bounds check AFTER barrier.
   */
  if (outX >= outputWidth || outY >= outputHeight || channel >= imageChannels) {
    return;
  }

  /*
   * This thread's convolution window begins at
   * its local output position * stride.
   */
  int localInputX = localX * stride;
  int localInputY = localY * stride;

  int accum = 0;

  /*
   * ------------------------------------------------
   * CONVOLUTION FROM LOCAL MEMORY
   * ------------------------------------------------
   */
  for (int y = 0; y < maskWidth; y++) {
    for (int x = 0; x < maskWidth; x++) {
      int localImageIndex = (localInputY + y) * tileWidth + (localInputX + x);

      int maskIndex = y * maskWidth + x;

      accum += localData[localImageIndex] * maskData[maskIndex];
    }
  }

  int outputIndex = (outY * outputWidth + outX) * imageChannels + channel;

  outputData[outputIndex] = accum;
}