#include <CL/cl.h>
#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <netinet/in.h>
#include <opencv2/core/ocl.hpp>
#include <opencv2/opencv.hpp>
#include <signal.h>
#include <sstream>
#include <sys/socket.h>
#include <termios.h>
#include <unistd.h>
#include <vector>

using namespace cv;
using namespace std;
using namespace std::chrono;

// Global running flag
atomic<bool> running(true);

// Terminal state
termios originalTerminalSettings;
int originalTerminalFlags = 0;
bool terminalConfigured = false;

// OpenCL Context Definitions
struct OpenCLContext {
  cl_platform_id platform = nullptr;
  cl_device_id device = nullptr;
  cl_context context = nullptr;
  cl_command_queue queue = nullptr;
  cl_program program = nullptr;
  cl_kernel kernel = nullptr; // bgr_to_gray_downscale_2x
  cl_kernel k_draw = nullptr; // draw_box_borders_gpu
  cl_mem d_src = nullptr;
  cl_mem d_dst = nullptr;
  cl_mem d_boxes = nullptr;
};

// CPU usage
struct CpuSnapshot {
  unsigned long long totalUser = 0;    // CPU time spent running normal user programs
  unsigned long long totalUserLow = 0; // CPU time spent running lower-priority user programs
  unsigned long long totalSys = 0;     // CPU time spent inside the Linux kernel doing system-level work
  unsigned long long totalIdle = 0;    // CPU idle time
};

// Benchmark statistics
struct BenchmarkStats {
  unsigned long long samples = 0;
  unsigned long long facesDetected = 0;
  unsigned long long eyesDetected = 0;

  double videoFpsSum = 0.0;
  double preprocessMsSum = 0.0;
  double gpuKernelMsSum = 0.0;
  double faceDetectMsSum = 0.0;
  double eyeDetectMsSum = 0.0;
  double detectionMsSum = 0.0;
  double drawBoxesMsSum = 0.0;
  double totalComputeMsSum = 0.0;
};

// Signal handler
void signalHandler(int) { running = false; }

// enables keyboard input for hotkeys (G to toggle CPU/GPU, Q to quit)
// change terminal settings. Note, need to restore this when done.
bool enableNonBlockingInput() {

  termios settings;
  tcgetattr(STDIN_FILENO, &settings);

  // Don't require Enter
  settings.c_lflag &= ~ICANON;
  tcsetattr(STDIN_FILENO, TCSANOW, &settings);

  // Don't wait if no key has been pressed
  int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
  fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);

  return true;
}

// reverts terminal adjustments to default when closing the program
void restoreTerminal() {
  if (!terminalConfigured) {
    return;
  }
  tcsetattr(STDIN_FILENO, TCSANOW, &originalTerminalSettings);
  fcntl(STDIN_FILENO, F_SETFL, originalTerminalFlags);
  terminalConfigured = false;
}

// CPU usage
// gets cpu time counters from /proc/stat and returns a snapshot of the current CPU usage
CpuSnapshot readCpuSnapshot() {
  CpuSnapshot snap;
  ifstream statFile("/proc/stat");
  string line;

  if (getline(statFile, line)) {
    if (line.substr(0, 3) == "cpu") {
      stringstream ss(line);
      string cpuLabel;
      ss >> cpuLabel >> snap.totalUser >> snap.totalUserLow >> snap.totalSys >> snap.totalIdle;
    }
  }

  return snap;
}

// cpu utilization calculation as percantage over change in time period. compare two snapshots
double calculateCpuUsage(const CpuSnapshot &prev, const CpuSnapshot &curr) {
  // sum up all ticks to get total cpu work time and idle time, then calculate the percentage of non-idle time
  unsigned long long prevTotal = prev.totalUser + prev.totalUserLow + prev.totalSys + prev.totalIdle;
  unsigned long long currTotal = curr.totalUser + curr.totalUserLow + curr.totalSys + curr.totalIdle;
  unsigned long long totalDelta = currTotal - prevTotal;
  unsigned long long idleDelta = curr.totalIdle - prev.totalIdle;

  // Avoid division by zero if the CPU counters haven't changed
  if (totalDelta == 0) {
    return 0.0;
  }
  // CPU usage percentage is calculated as the proportion of time spent doing work (non-idle) over the total time elapsed
  return (1.0 - static_cast<double>(idleDelta) / static_cast<double>(totalDelta)) * 100.0;
}

// GPU usage
// reads qualcomm adreno GPU utilization from sysfs files. tries the cycle counter first, then falls back to the percentage file if the cycle counter is not available
double readGpuUsage() {
  // Read GPU busy cycles and total cycles from sysfs
  ifstream gpubusyFile("/sys/class/kgsl/kgsl-3d0/gpubusy");
  if (gpubusyFile.is_open()) {
    unsigned long long busyCycles = 0;
    unsigned long long totalCycles = 0;

    gpubusyFile >> busyCycles >> totalCycles;
    if (totalCycles > 0) {
      return static_cast<double>(busyCycles) / static_cast<double>(totalCycles) * 100.0;
    }
  }

  // read percentage from sysfs as a fallback if the cycle counter is not available
  ifstream altGpuFile("/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage");
  if (altGpuFile.is_open()) {
    double pct = 0.0;
    altGpuFile >> pct;
    return pct;
  }

  return -1.0;
}

// Initialize OpenCL, compiles kernel.cl, creates queues, buffers, and kernels for GPU processing
bool initOpenCL(OpenCLContext &ocl, const string &kernelFile, int width, int height) {
  cl_int err;
  cl_uint numPlatforms = 0;

  // located opencl platform
  err = clGetPlatformIDs(1, &ocl.platform, &numPlatforms);
  if (err != CL_SUCCESS || numPlatforms == 0) {
    cerr << "Could not find OpenCL platform\n";
    return false;
  }

  // query for physial gpu
  err = clGetDeviceIDs(ocl.platform, CL_DEVICE_TYPE_GPU, 1, &ocl.device, nullptr);
  if (err != CL_SUCCESS) {
    cerr << "Could not find GPU OpenCL device\n";
    return false;
  }

  // create context
  ocl.context = clCreateContext(nullptr, 1, &ocl.device, nullptr, nullptr, &err);
  if (err != CL_SUCCESS) {
    cerr << "Could not create OpenCL context\n";
    return false;
  }

  // create command queue with profiling enabled for measuring kernel execution time
  ocl.queue = clCreateCommandQueue(ocl.context, ocl.device, CL_QUEUE_PROFILING_ENABLE, &err);
  if (err != CL_SUCCESS) {
    cerr << "Could not create OpenCL queue\n";
    return false;
  }

  // load .cl kernel source file
  ifstream file(kernelFile);
  if (!file.is_open()) {
    cerr << "Could not open " << kernelFile << "\n";
    return false;
  }
  string srcStr((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
  const char *src = srcStr.c_str();
  size_t length = srcStr.length();

  // create program from source
  ocl.program = clCreateProgramWithSource(ocl.context, 1, &src, &length, &err);
  if (err != CL_SUCCESS) {
    cerr << "Could not create OpenCL program\n";
    return false;
  }

  // build program with optimization flags for performance
  err = clBuildProgram(ocl.program, 1, &ocl.device, "-cl-fast-relaxed-math", nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t logSize = 0;
    clGetProgramBuildInfo(ocl.program, ocl.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
    vector<char> log(logSize);
    clGetProgramBuildInfo(ocl.program, ocl.device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
    cerr << "OpenCL Build Error:\n" << log.data() << "\n";
    return false;
  }

  // create kernel for grayscale + downscale
  ocl.kernel = clCreateKernel(ocl.program, "bgr_to_gray_downscale_2x", &err);
  if (err != CL_SUCCESS) {
    cerr << "Could not create OpenCL kernel bgr_to_gray_downscale_2x\n";
    return false;
  }

  // create kernel for drawing bounding box borders
  ocl.k_draw = clCreateKernel(ocl.program, "draw_box_borders_gpu", &err);
  if (err != CL_SUCCESS) {
    cerr << "Could not create OpenCL kernel draw_box_borders_gpu\n";
    return false;
  }

  // allocate GPU buffers for source image, downscaled grayscale image, and bounding box data
  size_t srcBytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 3;
  size_t dstBytes = static_cast<size_t>(width / 2) * static_cast<size_t>(height / 2);

  // CL_MEM_READ_WRITE allows the GPU to read the camera image AND write bounding box borders to it
  ocl.d_src = clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, srcBytes, nullptr, &err);
  if (err != CL_SUCCESS) {
    cerr << "Could not create OpenCL source buffer\n";
    return false;
  }

  // CL_MEM_WRITE_ONLY because the downscaled grayscale image is only written by the GPU kernel and read back by the CPU for detection
  ocl.d_dst = clCreateBuffer(ocl.context, CL_MEM_WRITE_ONLY, dstBytes, nullptr, &err);
  if (err != CL_SUCCESS) {
    cerr << "Could not create OpenCL destination buffer\n";
    return false;
  }

  // each box has 8 ints: [x, y, width, height, valid, B, G, R]
  // 5 features (Face, 2 Eyes, 2 Ears) * 8 ints = 40 ints
  // allocates storage for face, left eye, right eye, left ear, right ear bounding boxes and their colors (BGR)
  ocl.d_boxes = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY, sizeof(int) * 40, nullptr, &err);
  if (err != CL_SUCCESS) {
    cerr << "Could not create OpenCL box buffer\n";
    return false;
  }

  return true;
}

// Cleanup OpenCL and release all resources
void cleanupOpenCL(OpenCLContext &ocl) {
  if (ocl.d_boxes)
    clReleaseMemObject(ocl.d_boxes);
  if (ocl.d_src)
    clReleaseMemObject(ocl.d_src);
  if (ocl.d_dst)
    clReleaseMemObject(ocl.d_dst);
  if (ocl.k_draw)
    clReleaseKernel(ocl.k_draw);
  if (ocl.kernel)
    clReleaseKernel(ocl.kernel);
  if (ocl.program)
    clReleaseProgram(ocl.program);
  if (ocl.queue)
    clReleaseCommandQueue(ocl.queue);
  if (ocl.context)
    clReleaseContext(ocl.context);
}

// downscale using opencv functions on CPU. returns the time taken in milliseconds
double processCpu(const Mat &frame, Mat &gray, Mat &smallGray, int smallWidth, int smallHeight) {
  auto start = high_resolution_clock::now();
  cvtColor(frame, gray, COLOR_BGR2GRAY);
  resize(gray, smallGray, Size(smallWidth, smallHeight), 0, 0, INTER_LINEAR);
  auto end = high_resolution_clock::now();

  return duration_cast<microseconds>(end - start).count() / 1000.0;
}

// transfers camera frame to the GPU, executes the kernel, and reads back the downscaled grayscale image for detection
bool processGpu(OpenCLContext &ocl, const Mat &frame, Mat &smallGray, int fullWidth, int fullHeight, int smallWidth, int smallHeight, double &kernelMs, double &endToEndMs) {
  auto gpuStart = high_resolution_clock::now();
  cl_int err;

  // write the full camera frame to the GPU buffer
  size_t sourceBytes = static_cast<size_t>(fullHeight) * frame.step; // number of bytes occupied by one row of the OpenCV Mat, frame.step is 3 pixels because BGR
  err = clEnqueueWriteBuffer(ocl.queue, ocl.d_src, CL_FALSE, 0, sourceBytes, frame.data, 0, nullptr, nullptr);
  if (err != CL_SUCCESS)
    return false;

  // fetch the step sizes (row stride) for the source and destination images to pass to the kernel
  int srcStep = static_cast<int>(frame.step);     // BGR
  int dstStep = static_cast<int>(smallGray.step); // downsampled size

  // bind kernel arguments for the grayscale + downscale kernel
  err |= clSetKernelArg(ocl.kernel, 0, sizeof(cl_mem), &ocl.d_src);
  err |= clSetKernelArg(ocl.kernel, 1, sizeof(cl_mem), &ocl.d_dst);
  err |= clSetKernelArg(ocl.kernel, 2, sizeof(int), &fullWidth);
  err |= clSetKernelArg(ocl.kernel, 3, sizeof(int), &fullHeight);
  err |= clSetKernelArg(ocl.kernel, 4, sizeof(int), &srcStep);
  err |= clSetKernelArg(ocl.kernel, 5, sizeof(int), &dstStep);
  if (err != CL_SUCCESS)
    return false;

  // define the global and local work sizes for the kernel execution
  size_t globalWorkSize[2] = {static_cast<size_t>(smallWidth), static_cast<size_t>(smallHeight)};
  // group threads into 16x16 blocks
  size_t localWorkSize[2] = {16, 16};
  cl_event kernelEvent = nullptr;

  // enqueue kernel for execution on GPU
  err = clEnqueueNDRangeKernel(ocl.queue, ocl.kernel, 2, nullptr, globalWorkSize, localWorkSize, 0, nullptr, &kernelEvent);
  if (err != CL_SUCCESS)
    return false;

  // Read back small image so OpenCV GPU/CPU cascade can evaluate it
  err = clEnqueueReadBuffer(ocl.queue, ocl.d_dst, CL_TRUE, 0, static_cast<size_t>(smallWidth) * static_cast<size_t>(smallHeight), smallGray.data, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    clReleaseEvent(kernelEvent);
    return false;
  }

  // calculate full wall time from GPU transfer, kernel execution, and readback
  auto gpuEnd = high_resolution_clock::now();
  endToEndMs = duration_cast<microseconds>(gpuEnd - gpuStart).count() / 1000.0;

  // measure the GPU compute time
  // https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/clGetEventProfilingInfo.html‰
  cl_ulong kernelStart = 0, kernelEnd = 0;
  clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_START, sizeof(kernelStart), &kernelStart, nullptr);
  clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_END, sizeof(kernelEnd), &kernelEnd, nullptr);

  // core kernel execution time in milliseconds
  kernelMs = static_cast<double>(kernelEnd - kernelStart) * 1e-6;
  clReleaseEvent(kernelEvent);
  return true;
}

// GPU Bounding Box Drawing Kernel
// transfers bounding box data to the GPU, executes the kernel to draw boxes on the original frame, and reads back the annotated frame
bool processGpuDrawBoxes(OpenCLContext &ocl, Mat &frame, int fullWidth, int fullHeight, const int *boxData, int numBoxes, int thickness, double &drawMs) {
  auto start = high_resolution_clock::now();
  cl_int err;

  // Upload box coordinates and colors to GPU buffer
  err = clEnqueueWriteBuffer(ocl.queue, ocl.d_boxes, CL_FALSE, 0, sizeof(int) * numBoxes * 8, boxData, 0, nullptr, nullptr);
  if (err != CL_SUCCESS)
    return false;

  // fetch the step size (row stride) for the source image to pass to the kernel
  int frameStep = static_cast<int>(frame.step); // BGR

  // bind kernel arguments for the draw boxes kernel
  err |= clSetKernelArg(ocl.k_draw, 0, sizeof(cl_mem), &ocl.d_src);
  err |= clSetKernelArg(ocl.k_draw, 1, sizeof(int), &fullWidth);  // original frame width
  err |= clSetKernelArg(ocl.k_draw, 2, sizeof(int), &fullHeight); // original frame height
  err |= clSetKernelArg(ocl.k_draw, 3, sizeof(int), &frameStep);
  err |= clSetKernelArg(ocl.k_draw, 4, sizeof(cl_mem), &ocl.d_boxes);
  err |= clSetKernelArg(ocl.k_draw, 5, sizeof(int), &numBoxes);
  err |= clSetKernelArg(ocl.k_draw, 6, sizeof(int), &thickness);
  if (err != CL_SUCCESS)
    return false;

  // define the global and local work sizes for the kernel execution
  size_t globalWorkSize[2] = {static_cast<size_t>(fullWidth), static_cast<size_t>(fullHeight)};
  size_t localWorkSize[2] = {16, 16};
  cl_event drawEvent = nullptr;

  // enqueue the draw boxes kernel for execution on the GPU
  err = clEnqueueNDRangeKernel(ocl.queue, ocl.k_draw, 2, nullptr, globalWorkSize, localWorkSize, 0, nullptr, &drawEvent);
  if (err != CL_SUCCESS)
    return false;

  // Read back annotated frame from GPU memory
  size_t frameBytes = static_cast<size_t>(fullHeight) * frame.step;
  err = clEnqueueReadBuffer(ocl.queue, ocl.d_src, CL_TRUE, 0, frameBytes, frame.data, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    clReleaseEvent(drawEvent);
    return false;
  }

  // wait for the draw kernel to finish and measure its execution time using the profiling event
  clWaitForEvents(1, &drawEvent);

  // calculate gpu runtime
  cl_ulong kStart = 0, kEnd = 0;
  clGetEventProfilingInfo(drawEvent, CL_PROFILING_COMMAND_START, sizeof(kStart), &kStart, nullptr);
  clGetEventProfilingInfo(drawEvent, CL_PROFILING_COMMAND_END, sizeof(kEnd), &kEnd, nullptr);
  drawMs = static_cast<double>(kEnd - kStart) * 1e-6;
  clReleaseEvent(drawEvent);

  auto end = high_resolution_clock::now();
  return true;
}

// Draw benchmark statistics panel
// draws a display on the video frame showing CPU/GPU usage, processing times, FPS, and detected face/eye counts
void drawBenchmarkOverlay(Mat &img, bool useGpu, double cpuUsage, double gpuUsage, double preprocessMs, double gpuKernelMs, double detectionMs, double drawBoxesMs, double totalComputeMs, double processingFps, double videoFps, int faceCount, int eyeCount) {
  vector<string> lines;
  stringstream ss;

  // Display the current processing mode (CPU or GPU)
  ss << "MODE: " << (useGpu ? "GPU (OpenCV API + GPU Draw)" : "CPU");
  lines.push_back(ss.str());

  // Display CPU and GPU usage percentages
  ss.str("");
  ss.clear();
  ss << fixed << setprecision(1) << "CPU: " << cpuUsage << "% | GPU: ";
  if (gpuUsage >= 0.0)
    ss << gpuUsage << "%";
  else
    ss << "N/A";
  lines.push_back(ss.str());

  // Display preprocessing time and GPU kernel execution time if using GPU
  ss.str("");
  ss.clear();
  ss << fixed << setprecision(2);
  if (useGpu) {
    ss << "Pre: " << preprocessMs << " ms (K: " << gpuKernelMs << " ms)";
  } else {
    ss << "Pre: " << preprocessMs << " ms";
  }
  lines.push_back(ss.str());

  // Display detection time
  ss.str("");
  ss.clear();
  ss << fixed << setprecision(2) << "Detect: " << detectionMs << " ms";
  lines.push_back(ss.str());

  // Display GPU draw time if using GPU
  if (useGpu) {
    ss.str("");
    ss.clear();
    ss << fixed << setprecision(2) << "GPU Draw: " << drawBoxesMs << " ms";
    lines.push_back(ss.str());
  }

  // Display total compute time
  ss.str("");
  ss.clear();
  ss << fixed << setprecision(2) << "Compute: " << totalComputeMs << " ms";
  lines.push_back(ss.str());

  // Display processing FPS and video FPS
  ss.str("");
  ss.clear();
  ss << fixed << setprecision(1) << "Proc FPS: " << processingFps << " | Video FPS: " << videoFps;
  lines.push_back(ss.str());

  // Display detected face and eye counts
  ss.str("");
  ss.clear();
  ss << "Faces: " << faceCount << " | Eyes: " << eyeCount;
  lines.push_back(ss.str());

  // Calculate the size of the HUD box based on the text lines
  int fontFace = FONT_HERSHEY_SIMPLEX;
  double fontScale = 0.5;
  int thickness = 1;
  int lineSpacing = 22;
  int margin = 12;
  int maxTextWidth = 0;

  // Determine the maximum text width among all lines to size the HUD box appropriately
  for (const auto &text : lines) {
    int baseline = 0;
    Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
    maxTextWidth = max(maxTextWidth, textSize.width);
  }

  // Calculate the dimensions and position of the HUD box
  int boxWidth = maxTextWidth + margin * 2;
  int boxHeight = static_cast<int>(lines.size()) * lineSpacing + margin;
  int boxX = img.cols - boxWidth - 10;
  int boxY = 10;

  // Draw a semi-transparent rectangle for the HUD background and overlay the text lines
  Rect hudRect(boxX, boxY, boxWidth, boxHeight);
  if (hudRect.x >= 0 && hudRect.y >= 0 && hudRect.x + hudRect.width <= img.cols && hudRect.y + hudRect.height <= img.rows) {
    Mat roi = img(hudRect);
    Mat overlay;
    roi.copyTo(overlay);
    rectangle(overlay, Rect(0, 0, boxWidth, boxHeight), Scalar(15, 15, 15), FILLED);
    addWeighted(overlay, 0.7, roi, 0.3, 0, roi);
    rectangle(img, hudRect, Scalar(80, 80, 80), 1);
  }

  // Draw each line of text within the HUD box with appropriate spacing
  int textY = boxY + margin + 12;
  for (const auto &text : lines) {
    putText(img, text, Point(boxX + margin, textY), fontFace, fontScale, Scalar(0, 255, 255), thickness, LINE_AA);
    textY += lineSpacing;
  }
}

// Send all bytes over socket
// ensures that all data is sent over the socket, handling partial sends and errors
bool sendAll(int socketFd, const void *data, size_t length) {
  const char *buffer = static_cast<const char *>(data);

  while (length > 0) {
    ssize_t sent = send(socketFd, buffer, length, MSG_NOSIGNAL); // If client disconnected, don't terminate my entire program.
    if (sent <= 0)
      return false;
    buffer += sent;
    length -= static_cast<size_t>(sent);
  }

  return true;
}

// Main Execution
int main(int argc, char *argv[]) {
  // bind signal handlers for graceful shutdown on SIGINT and SIGTERM
  signal(SIGINT, signalHandler);
  signal(SIGTERM, signalHandler);

  string serverIp = "localhost";
  int port = 5000;

  // parse command line arguments for server IP and port
  if (argc == 3) {
    serverIp = argv[1];
    try {
      port = stoi(argv[2]);
    } catch (...) {
      cerr << "Invalid port: " << argv[2] << "\n";
      return 1;
    }
  } else if (argc != 1) {
    cerr << "Usage: " << argv[0] << " [IP port]\n";
    cerr << "Default: localhost 5000\n";
    cerr << "Example: " << argv[0] << " 0.0.0.0 5000\n";
    return 1;
  }

  // validate port number range
  if (port < 1 || port > 65535) {
    cerr << "Port must be between 1 and 65535\n";
    return 1;
  }

  // Detect and configure OpenCV GPU OpenCL acceleration
  bool oclAvailable = cv::ocl::haveOpenCL();
  if (oclAvailable) {
    cout << "\n[OpenCV GPU API] OpenCL Acceleration is AVAILABLE.\n";
    cv::ocl::Device dev = cv::ocl::Device::getDefault();
    cout << "[OpenCV GPU API] Target Device: " << dev.name() << " | Vendor: " << dev.vendorName() << "\n";
    cv::ocl::setUseOpenCL(false); // Start in CPU mode
  } else {
    cout << "\n[OpenCV GPU API] OpenCL is not available. Using CPU.\n";
  }

  // configure terminal for non-blocking keyboard input to allow hotkeys for toggling CPU/GPU and quitting
  bool keyboardEnabled = enableNonBlockingInput();
  atexit(restoreTerminal);

  bool useGpu = false;

  // open realtime benchmark log file for writing, truncating any existing content
  ofstream benchmarkLog("benchmark.log", ios::out | ios::trunc);
  if (!benchmarkLog.is_open()) {
    cerr << "Could not open benchmark.log\n";
    return 1;
  }

  // display startup information and instructions for hotkeys
  cout << "\nStarting in CPU MODE\n";
  cout << "Benchmark metrics will be written to benchmark.log\n";
  if (keyboardEnabled) {
    cout << "Press G to toggle CPU/GPU (OpenCL Preprocess + GPU Detection + GPU Drawing)\n";
    cout << "Press Q to quit\n";
  }

  // configure GStreamer pipeline to capture video from the Qualcomm RB3 camera, converting it to BGR format for OpenCV processing
  string inputPipeline = "qtiqmmfsrc camera=0 ! "
                         "video/x-raw,format=NV12,width=1280,height=720,framerate=30/1 ! "
                         "videoconvert ! "
                         "video/x-raw,format=BGR ! "
                         "appsink drop=true sync=false";

  // open the camera using OpenCV's VideoCapture with the GStreamer pipeline
  VideoCapture cap(inputPipeline, CAP_GSTREAMER);
  if (!cap.isOpened()) {
    cerr << "Could not open RB3 camera\n";
    return 1;
  }
  cout << "RB3 camera opened\n";

  // declare classic OpenCV Haar Cascade classifiers for face and eye detection
  CascadeClassifier faceCascade;
  CascadeClassifier eyeCascade;
  string cascadePath = "/usr/share/opencv4/haarcascades/";

  // load the Haar Cascade XML files for face and eye detection
  if (!faceCascade.load(cascadePath + "haarcascade_frontalface_default.xml") || !eyeCascade.load(cascadePath + "haarcascade_eye.xml")) {
    cerr << "Could not load cascade classifiers\n";
    return 1;
  }

  // read the first frame from the camera to determine its resolution and initialize OpenCL resources
  Mat frame;
  if (!cap.read(frame) || frame.empty()) {
    cerr << "Could not read first camera frame\n";
    return 1;
  }

  // initialize frame dimensions for full resolution and downscaled grayscale image
  int fullWidth = frame.cols;
  int fullHeight = frame.rows;
  int smallWidth = fullWidth / 2;
  int smallHeight = fullHeight / 2;

  // display the camera resolution to the console
  cout << "Camera resolution: " << fullWidth << "x" << fullHeight << "\n";

  // initialize OpenCL context, compile kernels, and allocate buffers for GPU processing
  OpenCLContext ocl;
  if (!initOpenCL(ocl, "kernel.cl", fullWidth, fullHeight)) {
    cerr << "Failed to initialize OpenCL GPU pipeline\n";
    return 1;
  }
  cout << "OpenCL GPU pipeline initialized (Downsample and greyscale + GPU Box Drawing)\n";

  // create a TCP server socket to stream the processed video frames over HTTP
  int serverSocket = socket(AF_INET, SOCK_STREAM, 0);
  if (serverSocket < 0) {
    cerr << "Could not create server socket\n";
    cleanupOpenCL(ocl);
    return 1;
  }

  // force reuse of the address to avoid "Address already in use" errors when restarting the server quickly
  int reuse = 1;
  setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

  // bind the server socket to the specified IP address and port
  sockaddr_in serverAddress{};
  serverAddress.sin_family = AF_INET;
  serverAddress.sin_port = htons(port);
  if (serverIp == "localhost")
    serverAddress.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  else if (serverIp == "0.0.0.0")
    serverAddress.sin_addr.s_addr = htonl(INADDR_ANY);
  else
    inet_pton(AF_INET, serverIp.c_str(), &serverAddress.sin_addr);

  // retry binding the server socket if the port is already in use, waiting 1 second between attempts
  while (running && bind(serverSocket, reinterpret_cast<sockaddr *>(&serverAddress), sizeof(serverAddress)) < 0) {
    cerr << "Could not bind to " << serverIp << ":" << port << ". Retrying...\n";
    sleep(1);
  }

  // if the server is no longer running exit
  if (!running) {
    close(serverSocket);
    cleanupOpenCL(ocl);
    return 0;
  }

  // listen for incoming connections on the server socket, allowing a backlog of 1 connection
  if (listen(serverSocket, 1) < 0) {
    cerr << "Could not listen on port " << port << "\n";
    close(serverSocket);
    cleanupOpenCL(ocl);
    return 1;
  }

  cout << "\nVideo available at:\n";
  cout << "http://" << serverIp << ":" << port << "\n";

  // set the server socket to non-blocking mode so that accept() does not block the main processing loop
  int serverFlags = fcntl(serverSocket, F_GETFL, 0);
  if (serverFlags < 0 || fcntl(serverSocket, F_SETFL, serverFlags | O_NONBLOCK) < 0) {
    perror("fcntl server socket");
    close(serverSocket);
    cleanupOpenCL(ocl);
    return 1;
  }

  int clientSocket = -1;

  // response header for streaming multipart JPEG frames to the client browser
  const string httpHeader = "HTTP/1.1 200 OK\r\n"
                            "Cache-Control: no-cache, no-store, must-revalidate\r\n"
                            "Pragma: no-cache\r\n"
                            "Expires: 0\r\n"
                            "Connection: close\r\n"
                            "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";

  // allocate OpenCV matrices for grayscale conversion and downscaled image
  Mat gray;
  Mat smallGray(smallHeight, smallWidth, CV_8UC1);
  const double invScale = 2.0;

  // baseline CPU snapshot for calculating CPU usage percentage over time
  CpuSnapshot lastCpuSnap = readCpuSnapshot();
  auto lastBenchmarkTime = high_resolution_clock::now();
  auto lastCaptureTimestamp = high_resolution_clock::now();

  // initialize variables for tracking current benchmark metrics
  double currentCpuPct = 0.0;
  double currentGpuPct = 0.0;
  double currentVideoFps = 0.0;
  double currentPreprocessMs = 0.0;
  double currentGpuKernelMs = 0.0;
  double currentFaceDetectMs = 0.0;
  double currentEyeDetectMs = 0.0;
  double currentDetectionMs = 0.0;
  double currentDrawBoxesMs = 0.0;
  double currentTotalComputeMs = 0.0;
  double currentProcessingFps = 0.0;
  double currentDetectionFps = 0.0;

  // counters for detected faces and eyes in the current frame
  int currentFaceCount = 0;
  int currentEyeCount = 0;

  // Benchmark statistics for CPU and GPU processing times
  BenchmarkStats cpuStats;
  BenchmarkStats gpuStats;

  // frame counter for calculating FPS and logging metrics
  int frameCount = 0;

  // log the start of the live benchmark to the benchmark log file
  benchmarkLog << "================ LIVE BENCHMARK ================\n";
  benchmarkLog << "Accelerated using Custom OpenCL and OpenCV GPU API (cv::ocl)\n";
  benchmarkLog << "MODE: CPU\n";
  benchmarkLog.flush();

  // Processing Loop
  while (running) {
    if (keyboardEnabled) {
      char key;
      // check for keyboard input to toggle CPU/GPU mode or quit the application
      while (read(STDIN_FILENO, &key, 1) > 0) {
        // Toggle CPU/GPU mode on 'G' key press
        if (key == 'g' || key == 'G') {
          useGpu = !useGpu;
          if (oclAvailable) {
            cv::ocl::setUseOpenCL(useGpu); // Enable/Disable OpenCV GPU OpenCL pipeline
          }
          // log the mode change to the console and benchmark log file
          cout << "\n====================================\n";
          cout << "MODE CHANGED TO: " << (useGpu ? "GPU (GPU Detection + GPU Drawing)" : "CPU") << "\n";
          cout << "====================================\n";
          benchmarkLog << "\nMODE CHANGED TO: " << (useGpu ? "GPU" : "CPU") << "\n";
          benchmarkLog.flush();
        } else if (key == 'q' || key == 'Q') {
          running = false;
          break;
        }
      }
    }

    // Capture a frame from the camera
    if (!running)
      break;

    // read a frame from the camera and check for errors
    if (!cap.read(frame) || frame.empty()) {
      cerr << "Frame capture error\n";
      break;
    }

    // benchmark the time taken to capture the frame and calculate the video FPS
    auto captureTimestamp = high_resolution_clock::now();
    double captureDeltaMs = duration_cast<microseconds>(captureTimestamp - lastCaptureTimestamp).count() / 1000.0;
    lastCaptureTimestamp = captureTimestamp;
    currentVideoFps = captureDeltaMs > 0.0 ? 1000.0 / captureDeltaMs : 0.0;

    // Preprocessing for OpenCL on GPU vs. OpenCV on CPU
    currentGpuKernelMs = 0.0;
    // Preprocess the frame using GPU or CPU based on the current mode
    if (useGpu) {
      double gpuEndToEndMs = 0.0;
      // Use OpenCL to preprocess the frame on the GPU
      if (!processGpu(ocl, frame, smallGray, fullWidth, fullHeight, smallWidth, smallHeight, currentGpuKernelMs, gpuEndToEndMs)) {
        cerr << "GPU preprocessing failed\n";
        break;
      }
      currentPreprocessMs = gpuEndToEndMs;
    } else {
      currentPreprocessMs = processCpu(frame, gray, smallGray, smallWidth, smallHeight);
    }
    // Face Detection
    vector<Rect> faces;
    auto faceDetectStart = high_resolution_clock::now();

    // Use OpenCV's detectMultiScale for face detection, leveraging GPU acceleration if available and enabled
    if (useGpu && oclAvailable) {
      // GPU Mode: UMat container routes detection directly to the Adreno GPU
      UMat u_smallGray = smallGray.getUMat(ACCESS_READ);
      faceCascade.detectMultiScale(u_smallGray, faces, 1.1, 4, 0, Size(30, 30));
    } else {
      // CPU Mode: Evaluated across host CPU threads
      faceCascade.detectMultiScale(smallGray, faces, 1.1, 4, 0, Size(30, 30));
    }

    // Benchmark the time taken for face detection
    auto faceDetectEnd = high_resolution_clock::now();
    currentFaceDetectMs = duration_cast<microseconds>(faceDetectEnd - faceDetectStart).count() / 1000.0;

    currentEyeDetectMs = 0.0;
    currentFaceCount = static_cast<int>(faces.size());
    currentEyeCount = 0;
    currentDrawBoxesMs = 0.0;

    // Pack bounding box data for GPU drawing. Each box has 8 integers representing [x, y, width, height, valid flag, B, G, R]
    int gpuBoxData[40] = {0}; // 5 boxes * 8 values each
    int boxIndex = 0;

    // Prepare labels for drawing on the frame
    vector<pair<string, Point>> labelsToDraw;

    // Iterate over detected faces and perform eye detection within each face region
    for (const Rect &smallFace : faces) {
      Rect face(cvRound(smallFace.x * invScale), cvRound(smallFace.y * invScale), cvRound(smallFace.width * invScale), cvRound(smallFace.height * invScale));

      // Ensure the face rectangle is within the bounds of the original frame
      face &= Rect(0, 0, frame.cols, frame.rows);
      if (face.width <= 0 || face.height <= 0)
        continue;

      // face bounding box data for GPU drawing
      gpuBoxData[0] = face.x;
      gpuBoxData[1] = face.y;
      gpuBoxData[2] = face.width;
      gpuBoxData[3] = face.height;
      gpuBoxData[4] = 1;   // Valid
      gpuBoxData[5] = 0;   // B
      gpuBoxData[6] = 0;   // G
      gpuBoxData[7] = 255; // R
      labelsToDraw.push_back({useGpu ? "Face (GPU)" : "Face (CPU)", Point(face.x, max(0, face.y - 5))});

      // Eye Detection within the detected face region
      vector<Rect> eyes;
      auto eyeDetectStart = high_resolution_clock::now();

      // Use OpenCV's detectMultiScale for eye detection, leveraging GPU acceleration if available and enabled
      if (useGpu && oclAvailable) {
        // Eye detection on GPU via OpenCV OpenCL UMat
        UMat u_faceROI = smallGray(smallFace).getUMat(ACCESS_READ);
        eyeCascade.detectMultiScale(u_faceROI, eyes, 1.1, 4, 0, Size(15, 15));
      } else {
        // Eye detection on CPU
        Mat faceROI = smallGray(smallFace);
        eyeCascade.detectMultiScale(faceROI, eyes, 1.1, 4, 0, Size(15, 15));
      }

      // Benchmark the time taken for eye detection
      auto eyeDetectEnd = high_resolution_clock::now();
      currentEyeDetectMs += duration_cast<microseconds>(eyeDetectEnd - eyeDetectStart).count() / 1000.0;
      currentEyeCount = static_cast<int>(eyes.size());

      // Iterate over detected eyes and prepare bounding box data for GPU drawing
      for (const Rect &smallEye : eyes) {
        Rect eyeGlobal(cvRound((smallFace.x + smallEye.x) * invScale), cvRound((smallFace.y + smallEye.y) * invScale), cvRound(smallEye.width * invScale), cvRound(smallEye.height * invScale));

        eyeGlobal &= Rect(0, 0, frame.cols, frame.rows);
        if (eyeGlobal.width <= 0 || eyeGlobal.height <= 0)
          continue;

        // Determine if the detected eye is the right or left eye based on its position relative to the face center
        double eyeCenterX = smallEye.x + smallEye.width * 0.5;
        double faceCenterX = smallFace.width * 0.5;
        bool isRightEye = (eyeCenterX < faceCenterX);
        string eyeLabel = isRightEye ? "R Eye" : "L Eye";

        // eye bounding box data for GPU drawing
        int eyeSlot = isRightEye ? 1 : 2;
        gpuBoxData[eyeSlot * 8 + 0] = eyeGlobal.x;
        gpuBoxData[eyeSlot * 8 + 1] = eyeGlobal.y;
        gpuBoxData[eyeSlot * 8 + 2] = eyeGlobal.width;
        gpuBoxData[eyeSlot * 8 + 3] = eyeGlobal.height;
        gpuBoxData[eyeSlot * 8 + 4] = 1;   // Valid
        gpuBoxData[eyeSlot * 8 + 5] = 0;   // B
        gpuBoxData[eyeSlot * 8 + 6] = 255; // G
        gpuBoxData[eyeSlot * 8 + 7] = 0;   // R
        labelsToDraw.push_back({eyeLabel, Point(eyeGlobal.x, max(0, eyeGlobal.y - 4))});
      }

      // Proportional ear location based on detected face rectangle, assuming ears are roughly 18% of face width and 35% of face height, positioned at 28% down from the top of the face
      int earWidth = static_cast<int>(face.width * 0.18);
      int earHeight = static_cast<int>(face.height * 0.35);
      int earY = face.y + static_cast<int>(face.height * 0.28);

      // Ensure ear rectangles are within the bounds of the original frame
      if (earWidth > 0 && earHeight > 0) {
        // Screen-left flank -> Subject Right Ear (Cyan: B=255, G=255, R=0)
        Rect rightEarRect(max(0, face.x - static_cast<int>(earWidth * 0.6)), earY, earWidth, earHeight);
        // Screen-right flank -> Subject Left Ear (Blue: B=255, G=0, R=0)
        Rect leftEarRect(min(frame.cols - earWidth, face.x + face.width - static_cast<int>(earWidth * 0.4)), earY, earWidth, earHeight);

        // Clip ear rectangles to ensure they are within the frame boundaries
        rightEarRect &= Rect(0, 0, frame.cols, frame.rows);
        leftEarRect &= Rect(0, 0, frame.cols, frame.rows);

        // Prepare bounding box data for detected ears for GPU drawing
        if (rightEarRect.width > 0 && rightEarRect.height > 0) {
          gpuBoxData[3 * 8 + 0] = rightEarRect.x;
          gpuBoxData[3 * 8 + 1] = rightEarRect.y;
          gpuBoxData[3 * 8 + 2] = rightEarRect.width;
          gpuBoxData[3 * 8 + 3] = rightEarRect.height;
          gpuBoxData[3 * 8 + 4] = 1;
          gpuBoxData[3 * 8 + 5] = 255; // B
          gpuBoxData[3 * 8 + 6] = 255; // G
          gpuBoxData[3 * 8 + 7] = 0;   // R
          labelsToDraw.push_back({"R Ear", Point(rightEarRect.x, max(0, rightEarRect.y - 4))});
        }

        // Prepare bounding box data for detected left ear for GPU drawing
        if (leftEarRect.width > 0 && leftEarRect.height > 0) {
          gpuBoxData[4 * 8 + 0] = leftEarRect.x;
          gpuBoxData[4 * 8 + 1] = leftEarRect.y;
          gpuBoxData[4 * 8 + 2] = leftEarRect.width;
          gpuBoxData[4 * 8 + 3] = leftEarRect.height;
          gpuBoxData[4 * 8 + 4] = 1;
          gpuBoxData[4 * 8 + 5] = 255; // B
          gpuBoxData[4 * 8 + 6] = 0;   // G
          gpuBoxData[4 * 8 + 7] = 0;   // R
          labelsToDraw.push_back({"L Ear", Point(leftEarRect.x, max(0, leftEarRect.y - 4))});
        }
      }
      break; // Process primary detected face
    }

    // Bounding box drawing on GPU OpenCL Kernel vs. CPU OpenCV
    if (useGpu) {
      // Execute the GPU bounding box drawing kernel in OpenCL
      processGpuDrawBoxes(ocl, frame, fullWidth, fullHeight, gpuBoxData, 5, 2, currentDrawBoxesMs);
    } else {
      // CPU bounding box drawing using OpenCV rectangle
      for (int i = 0; i < 5; ++i) {
        if (gpuBoxData[i * 8 + 4] == 1) {
          Rect r(gpuBoxData[i * 8 + 0], gpuBoxData[i * 8 + 1], gpuBoxData[i * 8 + 2], gpuBoxData[i * 8 + 3]);
          Scalar color(gpuBoxData[i * 8 + 5], gpuBoxData[i * 8 + 6], gpuBoxData[i * 8 + 7]);
          rectangle(frame, r, color, 2);
        }
      }
    }

    // Overlay text labels on each box
    for (const auto &lbl : labelsToDraw) {
      Scalar txtCol = (lbl.first.find("Ear") != string::npos) ? (lbl.first == "R Ear" ? Scalar(255, 255, 0) : Scalar(255, 0, 0)) : (lbl.first.find("Eye") != string::npos) ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
      putText(frame, lbl.first, lbl.second, FONT_HERSHEY_SIMPLEX, 0.45, txtCol, 1, LINE_AA);
    }

    // Calculate total detection time and processing FPS
    currentDetectionMs = currentFaceDetectMs + currentEyeDetectMs;
    currentTotalComputeMs = currentPreprocessMs + currentDetectionMs + currentDrawBoxesMs;
    currentProcessingFps = currentTotalComputeMs > 0.0 ? 1000.0 / currentTotalComputeMs : 0.0;
    currentDetectionFps = currentDetectionMs > 0.0 ? 1000.0 / currentDetectionMs : 0.0;

    // Update benchmark statistics
    BenchmarkStats &stats = useGpu ? gpuStats : cpuStats;
    stats.samples++;
    stats.facesDetected += static_cast<unsigned long long>(currentFaceCount);
    stats.eyesDetected += static_cast<unsigned long long>(currentEyeCount);
    stats.videoFpsSum += currentVideoFps;
    stats.preprocessMsSum += currentPreprocessMs;
    stats.gpuKernelMsSum += currentGpuKernelMs;
    stats.faceDetectMsSum += currentFaceDetectMs;
    stats.eyeDetectMsSum += currentEyeDetectMs;
    stats.detectionMsSum += currentDetectionMs;
    stats.drawBoxesMsSum += currentDrawBoxesMs;
    stats.totalComputeMsSum += currentTotalComputeMs;

    // Periodic logging every 0.5 seconds
    auto now = high_resolution_clock::now();
    double intervalSec = duration_cast<milliseconds>(now - lastBenchmarkTime).count() / 1000.0;
    if (intervalSec >= 0.5) {
      CpuSnapshot currentCpuSnap = readCpuSnapshot();
      currentCpuPct = calculateCpuUsage(lastCpuSnap, currentCpuSnap);
      currentGpuPct = readGpuUsage();
      lastCpuSnap = currentCpuSnap;
      lastBenchmarkTime = now;

      // Log benchmark metrics to the benchmark log file
      benchmarkLog << fixed << setprecision(1) << "\n[" << (useGpu ? "GPU (OpenCV API + GPU Draw)" : "CPU") << "] Video: " << currentVideoFps << " FPS | CPU: " << currentCpuPct << "% | GPU: ";
      if (currentGpuPct >= 0.0)
        benchmarkLog << currentGpuPct << "%\n";
      else
        benchmarkLog << "N/A\n";

      // Log preprocessing time and GPU kernel execution time if using GPU
      benchmarkLog << setprecision(2);
      if (useGpu) {
        benchmarkLog << "  Preprocess : " << currentPreprocessMs << " ms (kernel " << currentGpuKernelMs << " ms)\n"
                     << "  GPU Draw   : " << currentDrawBoxesMs << " ms\n";
      } else {
        benchmarkLog << "  Preprocess : " << currentPreprocessMs << " ms\n";
      }

      // Log detection time, face and eye detection times, and FPS metrics
      benchmarkLog << "  Detection  : " << currentDetectionMs << " ms"
                   << " | face " << currentFaceDetectMs << " ms"
                   << " | eyes " << currentEyeDetectMs << " ms"
                   << " | " << setprecision(1) << currentDetectionFps << " detect FPS\n"
                   << setprecision(2) << "  Compute    : " << currentTotalComputeMs << " ms/frame"
                   << " | " << setprecision(1) << currentProcessingFps << " processing FPS\n"
                   << "  Detected   : " << currentFaceCount << (currentFaceCount == 1 ? " face" : " faces") << " | " << currentEyeCount << (currentEyeCount == 1 ? " eye" : " eyes") << "\n";
      benchmarkLog.flush();
    }

    // Overlay top corner display
    drawBenchmarkOverlay(frame, useGpu, currentCpuPct, currentGpuPct, currentPreprocessMs, currentGpuKernelMs, currentDetectionMs, currentDrawBoxesMs, currentTotalComputeMs, currentProcessingFps, currentVideoFps, currentFaceCount, currentEyeCount);

    // Non-blocking Client Stream Handler
    if (clientSocket < 0) {
      sockaddr_in clientAddress{};
      socklen_t clientLength = sizeof(clientAddress);
      int newClientSocket = accept(serverSocket, reinterpret_cast<sockaddr *>(&clientAddress), &clientLength);
      if (newClientSocket >= 0) {
        // Set the new client socket to non-blocking mode to avoid blocking the main loop during send operations
        int clientFlags = fcntl(newClientSocket, F_GETFL, 0);
        if (clientFlags >= 0)
          fcntl(newClientSocket, F_SETFL, clientFlags | O_NONBLOCK);
        if (!sendAll(newClientSocket, httpHeader.data(), httpHeader.size())) {
          close(newClientSocket);
        } else {
          clientSocket = newClientSocket;
          cout << "Viewer connected\n";
        }
      } else if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
        perror("accept");
      }
    }

    // If a client is connected, encode the current frame as JPEG and send it over the socket
    if (clientSocket < 0) {
      frameCount++;
      continue;
    }

    // Encode the frame as JPEG with a quality of 85
    vector<uchar> jpeg;
    vector<int> jpegParams = {IMWRITE_JPEG_QUALITY, 85};
    if (!imencode(".jpg", frame, jpeg, jpegParams)) {
      cerr << "JPEG encoding failed\n";
      continue;
    }

    // Prepare the multipart HTTP frame header with the appropriate content length for the JPEG image
    string frameHeader = "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + to_string(jpeg.size()) + "\r\n\r\n";

    if (!sendAll(clientSocket, frameHeader.data(), frameHeader.size()) || !sendAll(clientSocket, jpeg.data(), jpeg.size()) || !sendAll(clientSocket, "\r\n", 2)) {
      cerr << "Viewer disconnected\n";
      close(clientSocket);
      clientSocket = -1;
      frameCount++;
      continue;
    }

    frameCount++;
  }

  // Final benchmark summary output
  auto printSummary = [](ostream &out, const string &label, const BenchmarkStats &stats, bool gpuMode) {
    if (stats.samples == 0)
      return;
    double sampleCount = static_cast<double>(stats.samples);
    double avgVideoFps = stats.videoFpsSum / sampleCount;
    double avgPreprocessMs = stats.preprocessMsSum / sampleCount;
    double avgKernelMs = stats.gpuKernelMsSum / sampleCount;
    double avgFaceDetectMs = stats.faceDetectMsSum / sampleCount;
    double avgEyeDetectMs = stats.eyeDetectMsSum / sampleCount;
    double avgDetectionMs = stats.detectionMsSum / sampleCount;
    double avgDrawBoxesMs = stats.drawBoxesMsSum / sampleCount;
    double avgTotalComputeMs = stats.totalComputeMsSum / sampleCount;
    double avgProcessingFps = avgTotalComputeMs > 0.0 ? 1000.0 / avgTotalComputeMs : 0.0;
    double avgDetectionFps = avgDetectionMs > 0.0 ? 1000.0 / avgDetectionMs : 0.0;
    double avgFaces = static_cast<double>(stats.facesDetected) / sampleCount;
    double avgEyes = static_cast<double>(stats.eyesDetected) / sampleCount;

    // Output the benchmark summary for the specified mode (CPU or GPU) with average metrics
    out << "\n---------------- " << label << " ----------------\n" << fixed << setprecision(2) << "Average preprocess      : " << avgPreprocessMs << " ms\n";
    if (gpuMode) {
      out << "Average GPU kernel      : " << avgKernelMs << " ms\n"
          << "Average GPU box draw    : " << avgDrawBoxesMs << " ms\n";
    }
    out << "Average face detection  : " << avgFaceDetectMs << " ms\n"
        << "Average eye detection   : " << avgEyeDetectMs << " ms\n"
        << "Average total detection : " << avgDetectionMs << " ms | " << setprecision(1) << avgDetectionFps << " detect FPS\n"
        << setprecision(2) << "Average total compute   : " << avgTotalComputeMs << " ms/frame | " << setprecision(1) << avgProcessingFps << " processing FPS\n"
        << "Average observed video  : " << avgVideoFps << " FPS\n"
        << setprecision(2) << "Average detections/frame: " << avgFaces << " faces | " << avgEyes << " eyes\n";
  };

  benchmarkLog << "\n================ FINAL BENCHMARK SUMMARY ================\n";
  printSummary(benchmarkLog, "CPU MODE", cpuStats, false);
  printSummary(benchmarkLog, "GPU MODE (OpenCV API + GPU Draw)", gpuStats, true);

  // Calculate and log speedup metrics if both CPU and GPU samples are available
  if (cpuStats.samples > 0 && gpuStats.samples > 0) {
    double avgCpuPreprocess = cpuStats.preprocessMsSum / static_cast<double>(cpuStats.samples);
    double avgGpuPreprocess = gpuStats.preprocessMsSum / static_cast<double>(gpuStats.samples);
    double avgCpuTotal = cpuStats.totalComputeMsSum / static_cast<double>(cpuStats.samples);
    double avgGpuTotal = gpuStats.totalComputeMsSum / static_cast<double>(gpuStats.samples);

    benchmarkLog << "\n---------------- SPEEDUP ----------------\n" << fixed << setprecision(2);
    if (avgGpuPreprocess > 0.0)
      benchmarkLog << "Preprocessing speedup   : " << avgCpuPreprocess / avgGpuPreprocess << "x\n";
    if (avgGpuTotal > 0.0)
      benchmarkLog << "Full compute speedup    : " << avgCpuTotal / avgGpuTotal << "x\n";
  }

  benchmarkLog << "=========================================================\n";
  benchmarkLog.flush();

  cout << "\nBenchmark complete. Results saved to benchmark.log\n";

  // Cleanup resources and restore terminal settings before exiting
  if (clientSocket >= 0)
    close(clientSocket);
  close(serverSocket);
  cap.release();
  cleanupOpenCL(ocl);
  restoreTerminal();

  return 0;
}