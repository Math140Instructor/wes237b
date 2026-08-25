#include <CL/cl.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#define CL_CHECK(err, what)                                                    \
  do {                                                                         \
    if ((err) != CL_SUCCESS) {                                                 \
      cerr << what << " failed with OpenCL error " << (err) << endl;           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

static const char *gpuSource = R"CLC(

__kernel void bgr_to_gray(
    __global const uchar *bgr,
    __global uchar *gray,
    const int width,
    const int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    int pixel = y * width + x;
    int i = pixel * 3;

    float b = (float)bgr[i + 0];
    float g = (float)bgr[i + 1];
    float r = (float)bgr[i + 2];

    gray[pixel] =
        convert_uchar_sat_rte(
            0.114f * b +
            0.587f * g +
            0.299f * r);
}

__kernel void alpha_blend_bgra(
    __global uchar *frame,
    const int frameWidth,
    const int frameHeight,

    __global const uchar *overlay,
    const int overlayWidth,
    const int overlayHeight,

    const int startX,
    const int startY)
{
    int gx = get_global_id(0);
    int gy = get_global_id(1);

    if (gx >= overlayWidth || gy >= overlayHeight)
        return;

    int fx = startX + gx;
    int fy = startY + gy;

    if (fx < 0 || fy < 0 ||
        fx >= frameWidth || fy >= frameHeight)
        return;

    int oi = (gy * overlayWidth + gx) * 4;
    int fi = (fy * frameWidth + fx) * 3;

    float alpha =
        ((float)overlay[oi + 3]) *
        (1.0f / 255.0f);

    float inv = 1.0f - alpha;

    frame[fi + 0] =
        convert_uchar_sat_rte(
            (float)overlay[oi + 0] * alpha +
            (float)frame[fi + 0] * inv);

    frame[fi + 1] =
        convert_uchar_sat_rte(
            (float)overlay[oi + 1] * alpha +
            (float)frame[fi + 1] * inv);

    frame[fi + 2] =
        convert_uchar_sat_rte(
            (float)overlay[oi + 2] * alpha +
            (float)frame[fi + 2] * inv);
}

)CLC";

class OpenCLGPU {
public:
  OpenCLGPU()
      : platform(nullptr), device(nullptr), context(nullptr),
        queue(nullptr), program(nullptr), grayKernel(nullptr),
        alphaKernel(nullptr), frameBuffer(nullptr),
        grayBuffer(nullptr), overlayBuffer(nullptr),
        frameBytes(0), grayBytes(0), overlayBytes(0) {}

  ~OpenCLGPU() {
    if (overlayBuffer)
      clReleaseMemObject(overlayBuffer);
    if (grayBuffer)
      clReleaseMemObject(grayBuffer);
    if (frameBuffer)
      clReleaseMemObject(frameBuffer);

    if (alphaKernel)
      clReleaseKernel(alphaKernel);
    if (grayKernel)
      clReleaseKernel(grayKernel);
    if (program)
      clReleaseProgram(program);
    if (queue)
      clReleaseCommandQueue(queue);
    if (context)
      clReleaseContext(context);
  }

  void initialize() {
    cl_int err;

    cl_uint platformCount = 0;
    err = clGetPlatformIDs(0, nullptr, &platformCount);
    CL_CHECK(err, "clGetPlatformIDs(count)");

    if (platformCount == 0) {
      cerr << "No OpenCL platforms found." << endl;
      exit(EXIT_FAILURE);
    }

    vector<cl_platform_id> platforms(platformCount);

    err = clGetPlatformIDs(platformCount, platforms.data(), nullptr);
    CL_CHECK(err, "clGetPlatformIDs");

    bool foundGPU = false;

    for (cl_platform_id p : platforms) {
      cl_uint deviceCount = 0;

      err = clGetDeviceIDs(
          p,
          CL_DEVICE_TYPE_GPU,
          0,
          nullptr,
          &deviceCount);

      if (err == CL_DEVICE_NOT_FOUND || deviceCount == 0)
        continue;

      CL_CHECK(err, "clGetDeviceIDs(count)");

      vector<cl_device_id> devices(deviceCount);

      err = clGetDeviceIDs(
          p,
          CL_DEVICE_TYPE_GPU,
          deviceCount,
          devices.data(),
          nullptr);

      CL_CHECK(err, "clGetDeviceIDs");

      platform = p;
      device = devices[0];
      foundGPU = true;
      break;
    }

    if (!foundGPU) {
      cerr << "No OpenCL GPU device found." << endl;
      exit(EXIT_FAILURE);
    }

    printDeviceInfo();

    context = clCreateContext(
        nullptr,
        1,
        &device,
        nullptr,
        nullptr,
        &err);

    CL_CHECK(err, "clCreateContext");

    queue = clCreateCommandQueue(
        context,
        device,
        CL_QUEUE_PROFILING_ENABLE,
        &err);

    CL_CHECK(err, "clCreateCommandQueue");

    size_t sourceLength = strlen(gpuSource);

    program = clCreateProgramWithSource(
        context,
        1,
        &gpuSource,
        &sourceLength,
        &err);

    CL_CHECK(err, "clCreateProgramWithSource");

    err = clBuildProgram(
        program,
        1,
        &device,
        nullptr,
        nullptr,
        nullptr);

    if (err != CL_SUCCESS) {
      size_t logSize = 0;

      clGetProgramBuildInfo(
          program,
          device,
          CL_PROGRAM_BUILD_LOG,
          0,
          nullptr,
          &logSize);

      vector<char> log(logSize);

      clGetProgramBuildInfo(
          program,
          device,
          CL_PROGRAM_BUILD_LOG,
          logSize,
          log.data(),
          nullptr);

      cerr << "OpenCL build failed:" << endl;
      cerr << log.data() << endl;
      exit(EXIT_FAILURE);
    }

    grayKernel = clCreateKernel(
        program,
        "bgr_to_gray",
        &err);

    CL_CHECK(err, "clCreateKernel(bgr_to_gray)");

    alphaKernel = clCreateKernel(
        program,
        "alpha_blend_bgra",
        &err);

    CL_CHECK(err, "clCreateKernel(alpha_blend_bgra)");

    cout << "OpenCL kernels compiled successfully." << endl;
  }

  void allocateFrameBuffers(int width, int height) {
    cl_int err;

    frameBytes =
        static_cast<size_t>(width) *
        static_cast<size_t>(height) *
        3;

    grayBytes =
        static_cast<size_t>(width) *
        static_cast<size_t>(height);

    frameBuffer = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE,
        frameBytes,
        nullptr,
        &err);

    CL_CHECK(err, "clCreateBuffer(frame)");

    grayBuffer = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE,
        grayBytes,
        nullptr,
        &err);

    CL_CHECK(err, "clCreateBuffer(gray)");
  }

  void bgrToGrayGPU(const Mat &frame, Mat &gray) {
    cl_int err;

    const int width = frame.cols;
    const int height = frame.rows;

    err = clEnqueueWriteBuffer(
        queue,
        frameBuffer,
        CL_TRUE,
        0,
        frameBytes,
        frame.data,
        0,
        nullptr,
        nullptr);

    CL_CHECK(err, "clEnqueueWriteBuffer(frame)");

    err = clSetKernelArg(
        grayKernel,
        0,
        sizeof(cl_mem),
        &frameBuffer);

    CL_CHECK(err, "clSetKernelArg(gray 0)");

    err = clSetKernelArg(
        grayKernel,
        1,
        sizeof(cl_mem),
        &grayBuffer);

    CL_CHECK(err, "clSetKernelArg(gray 1)");

    err = clSetKernelArg(
        grayKernel,
        2,
        sizeof(int),
        &width);

    CL_CHECK(err, "clSetKernelArg(gray 2)");

    err = clSetKernelArg(
        grayKernel,
        3,
        sizeof(int),
        &height);

    CL_CHECK(err, "clSetKernelArg(gray 3)");

    size_t global[2] = {
        static_cast<size_t>((width + 15) / 16 * 16),
        static_cast<size_t>((height + 15) / 16 * 16)
    };

    size_t local[2] = {16, 16};

    err = clEnqueueNDRangeKernel(
        queue,
        grayKernel,
        2,
        nullptr,
        global,
        local,
        0,
        nullptr,
        nullptr);

    CL_CHECK(err, "clEnqueueNDRangeKernel(bgr_to_gray)");

    gray.create(
        height,
        width,
        CV_8UC1);

    err = clEnqueueReadBuffer(
        queue,
        grayBuffer,
        CL_TRUE,
        0,
        grayBytes,
        gray.data,
        0,
        nullptr,
        nullptr);

    CL_CHECK(err, "clEnqueueReadBuffer(gray)");
  }

  void alphaBlendGPU(Mat &frame,
                     const Mat &overlay,
                     int startX,
                     int startY) {
    cl_int err;

    const int frameWidth = frame.cols;
    const int frameHeight = frame.rows;
    const int overlayWidth = overlay.cols;
    const int overlayHeight = overlay.rows;

    size_t newOverlayBytes =
        static_cast<size_t>(overlayWidth) *
        static_cast<size_t>(overlayHeight) *
        4;

    if (newOverlayBytes > overlayBytes) {
      if (overlayBuffer) {
        clReleaseMemObject(overlayBuffer);
        overlayBuffer = nullptr;
      }

      overlayBuffer = clCreateBuffer(
          context,
          CL_MEM_READ_ONLY,
          newOverlayBytes,
          nullptr,
          &err);

      CL_CHECK(err, "clCreateBuffer(overlay)");

      overlayBytes = newOverlayBytes;
    }

    err = clEnqueueWriteBuffer(
        queue,
        frameBuffer,
        CL_TRUE,
        0,
        frameBytes,
        frame.data,
        0,
        nullptr,
        nullptr);

    CL_CHECK(err, "clEnqueueWriteBuffer(alpha frame)");

    err = clEnqueueWriteBuffer(
        queue,
        overlayBuffer,
        CL_TRUE,
        0,
        newOverlayBytes,
        overlay.data,
        0,
        nullptr,
        nullptr);

    CL_CHECK(err, "clEnqueueWriteBuffer(overlay)");

    int arg = 0;

    err = clSetKernelArg(
        alphaKernel,
        arg++,
        sizeof(cl_mem),
        &frameBuffer);

    CL_CHECK(err, "clSetKernelArg(alpha frame)");

    err = clSetKernelArg(
        alphaKernel,
        arg++,
        sizeof(int),
        &frameWidth);

    CL_CHECK(err, "clSetKernelArg(frameWidth)");

    err = clSetKernelArg(
        alphaKernel,
        arg++,
        sizeof(int),
        &frameHeight);

    CL_CHECK(err, "clSetKernelArg(frameHeight)");

    err = clSetKernelArg(
        alphaKernel,
        arg++,
        sizeof(cl_mem),
        &overlayBuffer);

    CL_CHECK(err, "clSetKernelArg(overlay)");

    err = clSetKernelArg(
        alphaKernel,
        arg++,
        sizeof(int),
        &overlayWidth);

    CL_CHECK(err, "clSetKernelArg(overlayWidth)");

    err = clSetKernelArg(
        alphaKernel,
        arg++,
        sizeof(int),
        &overlayHeight);

    CL_CHECK(err, "clSetKernelArg(overlayHeight)");

    err = clSetKernelArg(
        alphaKernel,
        arg++,
        sizeof(int),
        &startX);

    CL_CHECK(err, "clSetKernelArg(startX)");

    err = clSetKernelArg(
        alphaKernel,
        arg++,
        sizeof(int),
        &startY);

    CL_CHECK(err, "clSetKernelArg(startY)");

    size_t global[2] = {
        static_cast<size_t>((overlayWidth + 15) / 16 * 16),
        static_cast<size_t>((overlayHeight + 15) / 16 * 16)
    };

    size_t local[2] = {16, 16};

    err = clEnqueueNDRangeKernel(
        queue,
        alphaKernel,
        2,
        nullptr,
        global,
        local,
        0,
        nullptr,
        nullptr);

    CL_CHECK(err, "clEnqueueNDRangeKernel(alpha_blend)");

    err = clEnqueueReadBuffer(
        queue,
        frameBuffer,
        CL_TRUE,
        0,
        frameBytes,
        frame.data,
        0,
        nullptr,
        nullptr);

    CL_CHECK(err, "clEnqueueReadBuffer(alpha frame)");
  }

private:
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_program program;
  cl_kernel grayKernel;
  cl_kernel alphaKernel;

  cl_mem frameBuffer;
  cl_mem grayBuffer;
  cl_mem overlayBuffer;

  size_t frameBytes;
  size_t grayBytes;
  size_t overlayBytes;

  void printDeviceInfo() {
    char name[256] = {};
    char vendor[256] = {};
    char version[256] = {};

    cl_uint computeUnits = 0;
    size_t maxWorkGroup = 0;

    clGetDeviceInfo(
        device,
        CL_DEVICE_NAME,
        sizeof(name),
        name,
        nullptr);

    clGetDeviceInfo(
        device,
        CL_DEVICE_VENDOR,
        sizeof(vendor),
        vendor,
        nullptr);

    clGetDeviceInfo(
        device,
        CL_DEVICE_VERSION,
        sizeof(version),
        version,
        nullptr);

    clGetDeviceInfo(
        device,
        CL_DEVICE_MAX_COMPUTE_UNITS,
        sizeof(computeUnits),
        &computeUnits,
        nullptr);

    clGetDeviceInfo(
        device,
        CL_DEVICE_MAX_WORK_GROUP_SIZE,
        sizeof(maxWorkGroup),
        &maxWorkGroup,
        nullptr);

    cout << "OpenCL GPU device: " << name << endl;
    cout << "OpenCL vendor: " << vendor << endl;
    cout << "OpenCL version: " << version << endl;
    cout << "Compute units: " << computeUnits << endl;
    cout << "Max work-group size: " << maxWorkGroup << endl;
  }
};

int main() {
  // =========================================================================
  // INITIALIZE REAL OPENCL GPU CONTEXT
  // =========================================================================

  OpenCLGPU gpu;
  gpu.initialize();

  // =========================================================================
  // CAMERA
  // =========================================================================

  string inputPipeline =
      "qtiqmmfsrc camera=0 ! "
      "video/x-raw,format=NV12,width=1280,height=720,framerate=30/1 ! "
      "videoconvert ! "
      "video/x-raw,format=BGR ! "
      "appsink drop=true sync=false";

  VideoCapture cap(
      inputPipeline,
      CAP_GSTREAMER);

  if (!cap.isOpened()) {
    cerr << "Could not open RB3 camera." << endl;
    return 1;
  }

  cout << "RB3 camera opened." << endl;

  // =========================================================================
  // HAAR CASCADES
  // =========================================================================

  CascadeClassifier faceCascade;
  CascadeClassifier eyeCascade;

  if (!faceCascade.load(
          "/usr/share/opencv4/haarcascades/"
          "haarcascade_frontalface_default.xml")) {
    cerr << "Could not load face cascade." << endl;
    return 1;
  }

  if (!eyeCascade.load(
          "/usr/share/opencv4/haarcascades/"
          "haarcascade_eye.xml")) {
    cerr << "Could not load eye cascade." << endl;
    return 1;
  }

  // =========================================================================
  // GLASSES
  // =========================================================================

  Mat glasses =
      imread(
          "glasses.png",
          IMREAD_UNCHANGED);

  if (glasses.empty()) {
    cerr << "Could not load glasses.png." << endl;
    return 1;
  }

  if (glasses.channels() != 4) {
    cerr << "glasses.png must have an alpha channel." << endl;
    return 1;
  }

  // =========================================================================
  // FIRST FRAME
  // =========================================================================

  Mat frame;

  if (!cap.read(frame) || frame.empty()) {
    cerr << "Could not read first camera frame." << endl;
    return 1;
  }

  if (!frame.isContinuous()) {
    frame = frame.clone();
  }

  cout << "Camera resolution: "
       << frame.cols
       << "x"
       << frame.rows
       << endl;

  gpu.allocateFrameBuffers(
      frame.cols,
      frame.rows);

  // =========================================================================
  // OUTPUT
  // =========================================================================

  string outputPipeline =
      "appsrc ! "
      "videoconvert ! "
      "x264enc tune=zerolatency ! "
      "video/x-h264,profile=baseline ! "
      "h264parse ! "
      "mp4mux ! "
      "filesink location=output.mp4";

  VideoWriter writer;

  writer.open(
      outputPipeline,
      CAP_GSTREAMER,
      0,
      30.0,
      frame.size(),
      true);

  if (!writer.isOpened()) {
    cerr << "Could not open output video pipeline." << endl;
    return 1;
  }

  cout << "Recording processed video to output.mp4" << endl;

  // =========================================================================
  // PROCESS
  // =========================================================================

  int frameCount = 0;
  int grayKernelLaunches = 0;
  int alphaKernelLaunches = 0;
  int faceFrames = 0;
  int twoEyeFrames = 0;

  const int maxFrames = 300;

  do {
    if (!frame.isContinuous()) {
      frame = frame.clone();
    }

    // -----------------------------------------------------------------------
    // GUARANTEED GPU WORK EVERY FRAME
    //
    // This is a direct clEnqueueNDRangeKernel() call.
    // -----------------------------------------------------------------------

    Mat gray;

    gpu.bgrToGrayGPU(
        frame,
        gray);

    grayKernelLaunches++;

    // -----------------------------------------------------------------------
    // CPU HAAR DETECTION
    // -----------------------------------------------------------------------

    vector<Rect> faces;

    faceCascade.detectMultiScale(
        gray,
        faces,
        1.1,
        5);

    if (!faces.empty()) {
      faceFrames++;
    }

    for (const Rect &face : faces) {
      Mat faceROI =
          gray(face);

      vector<Rect> eyes;

      eyeCascade.detectMultiScale(
          faceROI,
          eyes,
          1.1,
          5);

      if (eyes.size() < 2) {
        continue;
      }

      twoEyeFrames++;

      Point2f eye1(
          face.x +
              eyes[0].x +
              eyes[0].width / 2.0f,
          face.y +
              eyes[0].y +
              eyes[0].height / 2.0f);

      Point2f eye2(
          face.x +
              eyes[1].x +
              eyes[1].width / 2.0f,
          face.y +
              eyes[1].y +
              eyes[1].height / 2.0f);

      if (eye1.x > eye2.x) {
        swap(
            eye1,
            eye2);
      }

      double dx =
          eye2.x -
          eye1.x;

      double dy =
          eye2.y -
          eye1.y;

      double angle =
          atan2(
              dy,
              dx) *
          180.0 /
          CV_PI;

      double eyeDistance =
          sqrt(
              dx * dx +
              dy * dy);

      if (eyeDistance <= 1.0) {
        continue;
      }

      Point2f center(
          (eye1.x + eye2.x) / 2.0f,
          (eye1.y + eye2.y) / 2.0f);

      circle(
          frame,
          eye1,
          5,
          Scalar(0, 255, 0),
          -1);

      circle(
          frame,
          eye2,
          5,
          Scalar(0, 255, 0),
          -1);

      int width =
          static_cast<int>(
              eyeDistance *
              2.2);

      if (width <= 0) {
        continue;
      }

      int height =
          static_cast<int>(
              width *
              static_cast<double>(glasses.rows) /
              glasses.cols);

      if (height <= 0) {
        continue;
      }

      Mat resized;

      resize(
          glasses,
          resized,
          Size(width, height),
          0.0,
          0.0,
          INTER_LINEAR);

      Point2f glassCenter(
          resized.cols / 2.0f,
          resized.rows / 2.0f);

      Mat rotation =
          getRotationMatrix2D(
              glassCenter,
              angle,
              1.0);

      Mat rotated;

      warpAffine(
          resized,
          rotated,
          rotation,
          resized.size(),
          INTER_LINEAR,
          BORDER_CONSTANT,
          Scalar(0, 0, 0, 0));

      if (!rotated.isContinuous()) {
        rotated = rotated.clone();
      }

      int x =
          static_cast<int>(
              center.x -
              rotated.cols / 2.0f);

      int y =
          static_cast<int>(
              center.y -
              rotated.rows / 2.0f);

      // ---------------------------------------------------------------------
      // SECOND REAL GPU KERNEL:
      // glasses alpha blending
      // ---------------------------------------------------------------------

      gpu.alphaBlendGPU(
          frame,
          rotated,
          x,
          y);

      alphaKernelLaunches++;

      break;
    }

    writer.write(frame);

    frameCount++;

    if (frameCount % 30 == 0) {
      cout
          << "Processed " << frameCount
          << " frames"
          << " | GPU gray launches: "
          << grayKernelLaunches
          << " | GPU alpha launches: "
          << alphaKernelLaunches
          << " | frames with face: "
          << faceFrames
          << " | frames with 2 eyes: "
          << twoEyeFrames
          << endl;
    }

  } while (
      frameCount < maxFrames &&
      cap.read(frame) &&
      !frame.empty());

  writer.release();
  cap.release();

  cout << endl;
  cout << "Saved output.mp4" << endl;
  cout << "Total GPU grayscale launches: "
       << grayKernelLaunches
       << endl;
  cout << "Total GPU alpha-blend launches: "
       << alphaKernelLaunches
       << endl;

  return 0;
}