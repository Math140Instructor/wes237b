
__kernel void convolution2D(__global int *inputData, __global int *outputData, __constant int *maskData, //
                            int width, int height, int maskWidth, int imageChannels, int stride) {

  int outX = get_global_id(0);
  int outY = get_global_id(1);
  int channel = get_global_id(2);

  int outputWidth = (width - maskWidth) / stride + 1;
  int outputHeight = (height - maskWidth) / stride + 1;

  // Protect against extra work-items
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
}
