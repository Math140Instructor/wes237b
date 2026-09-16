#include <CL/cl.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace cv;
using namespace std;
using namespace std::chrono;

// Test-bench configuration

static int RECORD_SECONDS = 10; // default
static const int TARGET_CAMERA_FPS = 30;
static const int WARMUP_FRAMES = 30;
// static const int COOLDOWN_SECONDS = 0;
static const char *BENCHMARK_LOG = "benchmark.log";
static const char *PRIMARY_INPUT_VIDEO = "test_input.mp4";
static VideoCapture *g_camera = nullptr;

// OpenCL context

struct OpenCLContext {
  cl_platform_id platform = nullptr;
  cl_device_id device = nullptr;
  cl_context context = nullptr;
  cl_command_queue queue = nullptr;
  cl_program program = nullptr;
  cl_kernel kernel = nullptr;
  cl_mem d_src = nullptr;
  cl_mem d_dst = nullptr;
};

// CPU usage
struct CpuSnapshot {
  unsigned long long totalUser = 0;    // CPU time spent running normal user programs
  unsigned long long totalUserLow = 0; // CPU time spent running lower-priority user programs
  unsigned long long totalSys = 0;     // CPU time spent inside the Linux kernel doing system-level work
  unsigned long long totalIdle = 0;    // CPU idle time
};

CpuSnapshot readCpuSnapshot() {
  CpuSnapshot snap;
  ifstream statFile("/proc/stat");
  string line;

  if (getline(statFile, line) && line.substr(0, 3) == "cpu") {
    stringstream ss(line);
    string cpuLabel;
    ss >> cpuLabel >> snap.totalUser >> snap.totalUserLow >> snap.totalSys >> snap.totalIdle;
  }

  return snap;
}

double calculateCpuUsage(const CpuSnapshot &prev, const CpuSnapshot &curr) {
  unsigned long long prevTotal = prev.totalUser + prev.totalUserLow + prev.totalSys + prev.totalIdle;
  unsigned long long currTotal = curr.totalUser + curr.totalUserLow + curr.totalSys + curr.totalIdle;

  unsigned long long totalDelta = currTotal - prevTotal;
  unsigned long long idleDelta = curr.totalIdle - prev.totalIdle;

  if (totalDelta == 0) {
    return 0.0;
  }

  return (1.0 - static_cast<double>(idleDelta) / static_cast<double>(totalDelta)) * 100.0;
}

double readGpuUsage() {
  ifstream gpubusyFile("/sys/class/kgsl/kgsl-3d0/gpubusy");

  if (gpubusyFile.is_open()) {
    unsigned long long busyCycles = 0;
    unsigned long long totalCycles = 0;
    gpubusyFile >> busyCycles >> totalCycles;

    if (totalCycles > 0) {
      return static_cast<double>(busyCycles) / static_cast<double>(totalCycles) * 100.0;
    }
  }

  ifstream altGpuFile("/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage");

  if (altGpuFile.is_open()) {
    double pct = 0.0;
    altGpuFile >> pct;
    return pct;
  }

  return -1.0;
}

bool initOpenCL(OpenCLContext &ocl, const string &kernelFile, int width, int height) {
  cl_int err = CL_SUCCESS;
  cl_uint numPlatforms = 0;

  err = clGetPlatformIDs(1, &ocl.platform, &numPlatforms);
  if (err != CL_SUCCESS || numPlatforms == 0) {
    cerr << "Could not find OpenCL platform\n";
    return false;
  }

  err = clGetDeviceIDs(ocl.platform, CL_DEVICE_TYPE_GPU, 1, &ocl.device, nullptr);

  if (err != CL_SUCCESS) {
    cerr << "Could not find GPU OpenCL device\n";
    return false;
  }

  ocl.context = clCreateContext(nullptr, 1, &ocl.device, nullptr, nullptr, &err);

  if (err != CL_SUCCESS) {
    cerr << "Could not create OpenCL context\n";
    return false;
  }

  ocl.queue = clCreateCommandQueue(ocl.context, ocl.device, CL_QUEUE_PROFILING_ENABLE, &err);

  if (err != CL_SUCCESS) {
    cerr << "Could not create OpenCL command queue\n";
    return false;
  }

  ifstream file(kernelFile);
  if (!file.is_open()) {
    cerr << "Could not open " << kernelFile << "\n";
    return false;
  }

  string srcStr((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());

  const char *src = srcStr.c_str();
  size_t length = srcStr.length();

  ocl.program = clCreateProgramWithSource(ocl.context, 1, &src, &length, &err);

  if (err != CL_SUCCESS) {
    cerr << "Could not create OpenCL program\n";
    return false;
  }

  err = clBuildProgram(ocl.program, 1, &ocl.device, "-cl-fast-relaxed-math", nullptr, nullptr);

  if (err != CL_SUCCESS) {
    size_t logSize = 0;
    clGetProgramBuildInfo(ocl.program, ocl.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);

    vector<char> log(logSize);
    clGetProgramBuildInfo(ocl.program, ocl.device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);

    cerr << "OpenCL Build Error:\n" << log.data() << "\n";
    return false;
  }

  ocl.kernel = clCreateKernel(ocl.program, "bgr_to_gray_downscale_2x", &err);

  if (err != CL_SUCCESS) {
    cerr << "Could not create OpenCL kernel\n";
    return false;
  }

  size_t srcBytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 3;
  size_t dstBytes = static_cast<size_t>(width / 2) * static_cast<size_t>(height / 2);

  ocl.d_src = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY, srcBytes, nullptr, &err);

  if (err != CL_SUCCESS) {
    cerr << "Could not create OpenCL source buffer\n";
    return false;
  }

  ocl.d_dst = clCreateBuffer(ocl.context, CL_MEM_WRITE_ONLY, dstBytes, nullptr, &err);

  if (err != CL_SUCCESS) {
    cerr << "Could not create OpenCL destination buffer\n";
    return false;
  }

  return true;
}

void cleanupOpenCL(OpenCLContext &ocl) {
  if (ocl.d_src) {
    clReleaseMemObject(ocl.d_src);
    ocl.d_src = nullptr;
  }

  if (ocl.d_dst) {
    clReleaseMemObject(ocl.d_dst);
    ocl.d_dst = nullptr;
  }

  if (ocl.kernel) {
    clReleaseKernel(ocl.kernel);
    ocl.kernel = nullptr;
  }

  if (ocl.program) {
    clReleaseProgram(ocl.program);
    ocl.program = nullptr;
  }

  if (ocl.queue) {
    clReleaseCommandQueue(ocl.queue);
    ocl.queue = nullptr;
  }

  if (ocl.context) {
    clReleaseContext(ocl.context);
    ocl.context = nullptr;
  }
}

struct RecordingInfo {
  string path;
  int width = 0;
  int height = 0;
  int frames = 0;
  double fps = 0.0;
  double durationSec = 0.0;
};

bool openRecordingWriter(VideoWriter &writer, string &actualPath, int width, int height, double fps) {
  string hwPipeline = "appsrc ! videoconvert ! v4l2h264enc ! "
                      "video/x-h264,profile=baseline ! h264parse ! mp4mux ! "
                      "filesink location=" +
                      string(PRIMARY_INPUT_VIDEO);

  writer.open(hwPipeline, CAP_GSTREAMER, 0, fps, Size(width, height), true);

  if (writer.isOpened()) {
    actualPath = PRIMARY_INPUT_VIDEO;
    return true;
  }

  cerr << "Hardware H.264 recording failed; trying x264enc...\n";

  string swPipeline = "appsrc ! videoconvert ! x264enc tune=zerolatency speed-preset=ultrafast ! "
                      "video/x-h264,profile=baseline ! h264parse ! mp4mux ! "
                      "filesink location=" +
                      string(PRIMARY_INPUT_VIDEO);

  writer.open(swPipeline, CAP_GSTREAMER, 0, fps, Size(width, height), true);

  if (writer.isOpened()) {
    actualPath = PRIMARY_INPUT_VIDEO;
    return true;
  }

  return false;
}

bool recordTestVideo(RecordingInfo &info) {

  cout << "\n================ PHASE 1: RECORD TEST VIDEO ================\n";

  string inputPipeline = "qtiqmmfsrc camera=0 ! "
                         "video/x-raw,format=NV12,width=1280,height=720,framerate=30/1 ! "
                         "videoconvert ! "
                         "video/x-raw,format=BGR ! "
                         "appsink drop=true max-buffers=1 sync=false";

  // IMPORTANT:
  // Allocate the VideoCapture on the heap.
  //
  // Do NOT delete it and do NOT call release().
  // The Qualcomm qtiqmmfsrc pipeline hangs during OpenCV teardown.
  if (g_camera == nullptr) {
    g_camera = new VideoCapture();
  }

  cout << "Opening RB3 camera..." << endl;

  if (!g_camera->open(inputPipeline, CAP_GSTREAMER)) {
    cerr << "Could not open RB3 camera\n";
    return false;
  }

  if (!g_camera->isOpened()) {
    cerr << "RB3 camera is not open\n";
    return false;
  }

  cout << "Camera opened." << endl;

  Mat frame;

  cout << "Waiting for first frame..." << endl;

  if (!g_camera->read(frame) || frame.empty()) {
    cerr << "Could not read first camera frame\n";
    return false;
  }

  cout << "First frame received: " << frame.cols << "x" << frame.rows << endl;

  info.width = frame.cols;
  info.height = frame.rows;
  info.fps = TARGET_CAMERA_FPS;

  VideoWriter writer;

  if (!openRecordingWriter(writer, info.path, info.width, info.height, info.fps)) {

    cerr << "Could not create MP4 writer\n";
    return false;
  }

  const int targetFrames = RECORD_SECONDS * TARGET_CAMERA_FPS;
  cout << "Recording " << RECORD_SECONDS << " seconds (" << targetFrames << " frames) to " << info.path << "..." << endl;
  int frameCount = 0;

  auto start = steady_clock::now();

  while (frameCount < targetFrames) {
    if (!g_camera->read(frame) || frame.empty()) {
      cerr << "Camera capture failed at frame " << frameCount << endl;
      break;
    }
    writer.write(frame);
    frameCount++;
  }

  auto end = steady_clock::now();

  info.frames = frameCount;
  info.durationSec = duration_cast<milliseconds>(end - start).count() / 1000.0;

  cout << "Captured " << frameCount << "/" << targetFrames << " frames." << endl;
  cout << "Finalizing MP4..." << endl;
  writer.release();

  cout << "MP4 finalized." << endl;

  if (frameCount != targetFrames) {
    cerr << "Recording incomplete: expected " << targetFrames << " frames but captured " << frameCount << endl;
    return false;
  }

  cout << "Recorded " << info.frames << " frames in " << RECORD_SECONDS << " s" << endl;
  cout << "Saved input video: " << info.path << endl;
  cout << "Recording phase complete." << endl;

  return true;
}

double processCpu(const Mat &frame, Mat &gray, Mat &smallGray, int smallWidth, int smallHeight) {
  auto start = high_resolution_clock::now();

  cvtColor(frame, gray, COLOR_BGR2GRAY);
  resize(gray, smallGray, Size(smallWidth, smallHeight), 0, 0, INTER_LINEAR);

  auto end = high_resolution_clock::now();

  return duration_cast<microseconds>(end - start).count() / 1000.0;
}

bool processGpu(OpenCLContext &ocl, const Mat &inputFrame, Mat &smallGray, int fullWidth, int fullHeight, int smallWidth, int smallHeight, double &kernelMs, double &endToEndMs) {
  Mat contiguousFrame;
  const Mat *framePtr = &inputFrame;

  if (!inputFrame.isContinuous()) {
    contiguousFrame = inputFrame.clone();
    framePtr = &contiguousFrame;
  }

  const Mat &frame = *framePtr;

  auto gpuStart = high_resolution_clock::now();

  cl_int err = CL_SUCCESS;

  size_t sourceBytes = static_cast<size_t>(fullHeight) * frame.step;

  err = clEnqueueWriteBuffer(ocl.queue, ocl.d_src, CL_FALSE, 0, sourceBytes, frame.data, 0, nullptr, nullptr);

  if (err != CL_SUCCESS) {
    cerr << "clEnqueueWriteBuffer failed: " << err << "\n";
    return false;
  }

  int srcStep = static_cast<int>(frame.step);
  int dstStep = static_cast<int>(smallGray.step);

  err = CL_SUCCESS;
  err |= clSetKernelArg(ocl.kernel, 0, sizeof(cl_mem), &ocl.d_src);
  err |= clSetKernelArg(ocl.kernel, 1, sizeof(cl_mem), &ocl.d_dst);
  err |= clSetKernelArg(ocl.kernel, 2, sizeof(int), &fullWidth);
  err |= clSetKernelArg(ocl.kernel, 3, sizeof(int), &fullHeight);
  err |= clSetKernelArg(ocl.kernel, 4, sizeof(int), &srcStep);
  err |= clSetKernelArg(ocl.kernel, 5, sizeof(int), &dstStep);

  if (err != CL_SUCCESS) {
    cerr << "clSetKernelArg failed: " << err << "\n";
    return false;
  }

  size_t globalWorkSize[2] = {static_cast<size_t>(smallWidth), static_cast<size_t>(smallHeight)};

  size_t localWorkSize[2] = {16, 8};

  cl_event kernelEvent = nullptr;

  err = clEnqueueNDRangeKernel(ocl.queue, ocl.kernel, 2, nullptr, globalWorkSize, localWorkSize, 0, nullptr, &kernelEvent);

  if (err != CL_SUCCESS) {
    cerr << "clEnqueueNDRangeKernel failed: " << err << "\n";
    return false;
  }

  err = clEnqueueReadBuffer(ocl.queue, ocl.d_dst, CL_TRUE, 0, static_cast<size_t>(smallWidth) * static_cast<size_t>(smallHeight), smallGray.data, 0, nullptr, nullptr);

  if (err != CL_SUCCESS) {
    cerr << "clEnqueueReadBuffer failed: " << err << "\n";
    clReleaseEvent(kernelEvent);
    return false;
  }

  auto gpuEnd = high_resolution_clock::now();

  endToEndMs = duration_cast<microseconds>(gpuEnd - gpuStart).count() / 1000.0;

  cl_ulong kernelStart = 0;
  cl_ulong kernelEnd = 0;

  clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_START, sizeof(kernelStart), &kernelStart, nullptr);
  clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_END, sizeof(kernelEnd), &kernelEnd, nullptr);

  kernelMs = static_cast<double>(kernelEnd - kernelStart) * 1e-6;

  clReleaseEvent(kernelEvent);

  return true;
}

struct DetectionMetrics {
  double faceMs = 0.0;
  double eyeMs = 0.0;
  double earMs = 0.0;
  double drawMs = 0.0;
  int faceCount = 0;
  int eyeCount = 0;
};

DetectionMetrics runDetectionPipeline(const Mat &smallGray, Mat &displayFrame, CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade, double invScale) {
  DetectionMetrics metrics;

  vector<Rect> faces;

  auto faceStart = high_resolution_clock::now();
  faceCascade.detectMultiScale(smallGray, faces, 1.1, 4, 0, Size(30, 30));
  auto faceEnd = high_resolution_clock::now();

  metrics.faceMs = duration_cast<microseconds>(faceEnd - faceStart).count() / 1000.0;
  metrics.faceCount = static_cast<int>(faces.size());

  if (faces.empty()) {
    return metrics;
  }

  // Match the original program: only process the first detected face.
  const Rect &smallFace = faces.front();

  Rect face(cvRound(smallFace.x * invScale), cvRound(smallFace.y * invScale), cvRound(smallFace.width * invScale), cvRound(smallFace.height * invScale));

  face &= Rect(0, 0, displayFrame.cols, displayFrame.rows);

  if (face.width <= 0 || face.height <= 0) {
    return metrics;
  }

  auto drawStart = high_resolution_clock::now();

  rectangle(displayFrame, face, Scalar(0, 0, 255), 2);

  putText(displayFrame, "Face", Point(face.x, max(0, face.y - 5)), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);

  metrics.drawMs += duration_cast<microseconds>(high_resolution_clock::now() - drawStart).count() / 1000.0;

  Mat faceROI = smallGray(smallFace);
  vector<Rect> eyes;

  auto eyeStart = high_resolution_clock::now();
  eyeCascade.detectMultiScale(faceROI, eyes, 1.1, 4, 0, Size(15, 15));
  auto eyeEnd = high_resolution_clock::now();

  metrics.eyeMs = duration_cast<microseconds>(eyeEnd - eyeStart).count() / 1000.0;

  metrics.eyeCount = static_cast<int>(eyes.size());

  drawStart = high_resolution_clock::now();

  for (const Rect &smallEye : eyes) {
    Rect eyeGlobal(cvRound((smallFace.x + smallEye.x) * invScale), cvRound((smallFace.y + smallEye.y) * invScale), cvRound(smallEye.width * invScale), cvRound(smallEye.height * invScale));

    eyeGlobal &= Rect(0, 0, displayFrame.cols, displayFrame.rows);

    if (eyeGlobal.width <= 0 || eyeGlobal.height <= 0) {
      continue;
    }

    // Anatomical labeling from the subject's perspective:
    // image-left = subject's RIGHT eye, image-right = subject's LEFT eye.
    double eyeCenterX = smallEye.x + smallEye.width * 0.5;
    double faceCenterX = smallFace.width * 0.5;

    string eyeLabel = eyeCenterX < faceCenterX ? "R Eye" : "L Eye";

    rectangle(displayFrame, eyeGlobal, Scalar(0, 255, 0), 2);

    putText(displayFrame, eyeLabel, Point(eyeGlobal.x, max(0, eyeGlobal.y - 4)), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 255, 0), 1);
  }

  metrics.drawMs += duration_cast<microseconds>(high_resolution_clock::now() - drawStart).count() / 1000.0;

  auto earStart = high_resolution_clock::now();

  int earWidth = static_cast<int>(face.width * 0.18);

  int earHeight = static_cast<int>(face.height * 0.35);

  int earY = face.y + static_cast<int>(face.height * 0.28);

  Rect leftEarRect;
  Rect rightEarRect;

  if (earWidth > 0 && earHeight > 0) {
    leftEarRect = Rect(max(0, face.x - static_cast<int>(earWidth * 0.6)), earY, earWidth, earHeight);

    rightEarRect = Rect(min(displayFrame.cols - earWidth, face.x + face.width - static_cast<int>(earWidth * 0.4)), earY, earWidth, earHeight);

    leftEarRect &= Rect(0, 0, displayFrame.cols, displayFrame.rows);

    rightEarRect &= Rect(0, 0, displayFrame.cols, displayFrame.rows);
  }

  metrics.earMs = duration_cast<microseconds>(high_resolution_clock::now() - earStart).count() / 1000.0;

  drawStart = high_resolution_clock::now();

  if (leftEarRect.width > 0 && leftEarRect.height > 0) {
    rectangle(displayFrame, leftEarRect, Scalar(255, 0, 0), 2);

    putText(displayFrame, "L Ear", Point(leftEarRect.x, max(0, leftEarRect.y - 4)), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 0, 0), 1);
  }

  if (rightEarRect.width > 0 && rightEarRect.height > 0) {
    rectangle(displayFrame, rightEarRect, Scalar(255, 255, 0), 2);

    putText(displayFrame, "R Ear", Point(rightEarRect.x, max(0, rightEarRect.y - 4)), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 0), 1);
  }

  metrics.drawMs += duration_cast<microseconds>(high_resolution_clock::now() - drawStart).count() / 1000.0;

  return metrics;
}

enum class BenchmarkMode { CPU, GPU_ACCELERATED };

struct FrameMetrics {
  int frameIndex = 0;

  double preprocessMs = 0.0;
  double gpuKernelMs = 0.0;

  double faceMs = 0.0;
  double eyeMs = 0.0;
  double earMs = 0.0;
  double detectionMs = 0.0;
  double drawMs = 0.0;

  // Core compute excludes drawing.
  double computeMs = 0.0;

  // Full measured pipeline includes drawing.
  double pipelineMs = 0.0;

  double processingFps = 0.0;
  double pipelineFps = 0.0;

  int faces = 0;
  int eyes = 0;

  double cpuUsage = -1.0;
  double gpuUsage = -1.0;
};

struct PassResults {
  string label;
  BenchmarkMode mode = BenchmarkMode::CPU;

  int totalFramesRead = 0;
  int measuredFrames = 0;

  vector<FrameMetrics> frames;
  vector<double> cpuUsageSamples;
  vector<double> gpuUsageSamples;

  double wallTimeMs = 0.0;
};

double mean(const vector<double> &values) {
  if (values.empty()) {
    return 0.0;
  }

  return accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

double percentile(vector<double> values, double p) {
  if (values.empty()) {
    return 0.0;
  }

  sort(values.begin(), values.end());

  double position = (p / 100.0) * static_cast<double>(values.size() - 1);

  size_t lower = static_cast<size_t>(floor(position));
  size_t upper = static_cast<size_t>(ceil(position));

  if (lower == upper) {
    return values[lower];
  }

  double weight = position - static_cast<double>(lower);

  return values[lower] * (1.0 - weight) + values[upper] * weight;
}

vector<double> collectMetric(const vector<FrameMetrics> &frames, double FrameMetrics::*member) {
  vector<double> values;
  values.reserve(frames.size());

  for (const FrameMetrics &frame : frames) {
    values.push_back(frame.*member);
  }

  return values;
}

bool runBenchmarkPass(const string &inputVideo, BenchmarkMode mode, CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade, OpenCLContext *ocl, PassResults &results) {

  results.mode = mode;
  results.label = mode == BenchmarkMode::CPU ? "CPU" : "GPU-ACCELERATED (GPU preprocess + CPU Haar detection)";

  cout << "\n================ PHASE " << (mode == BenchmarkMode::CPU ? "2" : "3") << ": " << results.label << " ================\n";

  VideoCapture cap(inputVideo);

  if (!cap.isOpened()) {
    cerr << "Could not open recorded test video: " << inputVideo << "\n";
    return false;
  }

  int fullWidth = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
  int fullHeight = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));

  if (fullWidth <= 0 || fullHeight <= 0) {
    cerr << "Invalid recorded video dimensions\n";
    return false;
  }

  int smallWidth = fullWidth / 2;
  int smallHeight = fullHeight / 2;

  Mat frame;
  Mat gray;
  Mat smallGray(smallHeight, smallWidth, CV_8UC1);

  const double invScale = 2.0;

  CpuSnapshot lastCpuSnap = readCpuSnapshot();
  auto lastUtilTime = steady_clock::now();

  double currentCpuUsage = -1.0;
  double currentGpuUsage = -1.0;

  auto passStart = steady_clock::now();

  int frameIndex = 0;

  while (cap.read(frame)) {
    if (frame.empty()) {
      break;
    }

    FrameMetrics metrics;
    metrics.frameIndex = frameIndex;

    double gpuKernelMs = 0.0;

    if (mode == BenchmarkMode::CPU) {
      metrics.preprocessMs = processCpu(frame, gray, smallGray, smallWidth, smallHeight);

    } else {
      if (ocl == nullptr) {
        cerr << "GPU benchmark requested without OpenCL context\n";
        return false;
      }

      double gpuEndToEndMs = 0.0;

      if (!processGpu(*ocl, frame, smallGray, fullWidth, fullHeight, smallWidth, smallHeight, gpuKernelMs, gpuEndToEndMs)) {
        cerr << "GPU preprocessing failed on frame " << frameIndex << "\n";
        return false;
      }

      metrics.preprocessMs = gpuEndToEndMs;
      metrics.gpuKernelMs = gpuKernelMs;
    }

    // Drawing occurs on the decoded frame after preprocessing, so no extra
    // frame clone is needed or added to the benchmark workload.
    Mat &annotatedFrame = frame;

    DetectionMetrics detection = runDetectionPipeline(smallGray, annotatedFrame, faceCascade, eyeCascade, invScale);

    metrics.faceMs = detection.faceMs;
    metrics.eyeMs = detection.eyeMs;
    metrics.earMs = detection.earMs;
    metrics.drawMs = detection.drawMs;
    metrics.faces = detection.faceCount;
    metrics.eyes = detection.eyeCount;

    metrics.detectionMs = metrics.faceMs + metrics.eyeMs + metrics.earMs;

    metrics.computeMs = metrics.preprocessMs + metrics.detectionMs;

    metrics.pipelineMs = metrics.computeMs + metrics.drawMs;

    metrics.processingFps = metrics.computeMs > 0.0 ? 1000.0 / metrics.computeMs : 0.0;

    metrics.pipelineFps = metrics.pipelineMs > 0.0 ? 1000.0 / metrics.pipelineMs : 0.0;

    auto now = steady_clock::now();
    double utilIntervalSec = duration_cast<milliseconds>(now - lastUtilTime).count() / 1000.0;

    if (utilIntervalSec >= 0.5) {
      CpuSnapshot currentCpuSnap = readCpuSnapshot();
      currentCpuUsage = calculateCpuUsage(lastCpuSnap, currentCpuSnap);
      currentGpuUsage = readGpuUsage();

      lastCpuSnap = currentCpuSnap;
      lastUtilTime = now;

      if (frameIndex >= WARMUP_FRAMES) {
        results.cpuUsageSamples.push_back(currentCpuUsage);

        if (currentGpuUsage >= 0.0) {
          results.gpuUsageSamples.push_back(currentGpuUsage);
        }
      }
    }

    metrics.cpuUsage = currentCpuUsage;
    metrics.gpuUsage = currentGpuUsage;

    // Warm-up frames execute the full pipeline but are not used in results.
    if (frameIndex >= WARMUP_FRAMES) {
      results.frames.push_back(metrics);
    }

    frameIndex++;
  }

  auto passEnd = steady_clock::now();

  cap.release();

  results.totalFramesRead = frameIndex;
  results.measuredFrames = static_cast<int>(results.frames.size());
  results.wallTimeMs = duration_cast<milliseconds>(passEnd - passStart).count();

  cout << "Read " << results.totalFramesRead << " frames; " << results.measuredFrames << " measured after " << WARMUP_FRAMES << " warm-up frames.\n";

  return results.measuredFrames > 0;
}
// Log
void writeFrameTable(ofstream &log, const PassResults &results) {
  log << "\n================ PER-FRAME DATA: " << results.label << " ================\n";

  log << "frame,preprocess_ms,gpu_kernel_ms,face_ms,eye_ms,ear_ms,"
         "detection_ms,draw_ms,compute_ms,pipeline_ms,processing_fps,"
         "pipeline_fps,faces,eyes,cpu_pct,gpu_pct\n";

  log << fixed << setprecision(4);

  for (const FrameMetrics &f : results.frames) {
    log << f.frameIndex << "," << f.preprocessMs << "," << f.gpuKernelMs << "," << f.faceMs << "," << f.eyeMs << "," << f.earMs << "," << f.detectionMs << "," << f.drawMs << "," << f.computeMs << "," << f.pipelineMs << "," << f.processingFps << "," << f.pipelineFps << "," << f.faces << "," << f.eyes << "," << f.cpuUsage << "," << f.gpuUsage << "\n";
  }
}

struct SummaryNumbers {
  double avgPreprocess = 0.0;
  double medianPreprocess = 0.0;
  double p95Preprocess = 0.0;

  double avgGpuKernel = 0.0;

  double avgFace = 0.0;
  double avgEye = 0.0;
  double avgEar = 0.0;
  double avgDetection = 0.0;
  double avgDraw = 0.0;

  double avgCompute = 0.0;
  double medianCompute = 0.0;
  double p95Compute = 0.0;

  double avgPipeline = 0.0;
  double p95Pipeline = 0.0;

  double processingFps = 0.0;
  double pipelineFps = 0.0;

  double avgCpuUsage = 0.0;
  double avgGpuUsage = -1.0;

  long long totalFaces = 0;
  long long totalEyes = 0;
};

SummaryNumbers summarize(const PassResults &results) {
  SummaryNumbers s;

  vector<double> preprocess = collectMetric(results.frames, &FrameMetrics::preprocessMs);
  vector<double> kernel = collectMetric(results.frames, &FrameMetrics::gpuKernelMs);
  vector<double> face = collectMetric(results.frames, &FrameMetrics::faceMs);
  vector<double> eye = collectMetric(results.frames, &FrameMetrics::eyeMs);
  vector<double> ear = collectMetric(results.frames, &FrameMetrics::earMs);
  vector<double> detection = collectMetric(results.frames, &FrameMetrics::detectionMs);
  vector<double> draw = collectMetric(results.frames, &FrameMetrics::drawMs);
  vector<double> compute = collectMetric(results.frames, &FrameMetrics::computeMs);
  vector<double> pipeline = collectMetric(results.frames, &FrameMetrics::pipelineMs);

  s.avgPreprocess = mean(preprocess);
  s.medianPreprocess = percentile(preprocess, 50.0);
  s.p95Preprocess = percentile(preprocess, 95.0);

  s.avgGpuKernel = mean(kernel);

  s.avgFace = mean(face);
  s.avgEye = mean(eye);
  s.avgEar = mean(ear);
  s.avgDetection = mean(detection);
  s.avgDraw = mean(draw);

  s.avgCompute = mean(compute);
  s.medianCompute = percentile(compute, 50.0);
  s.p95Compute = percentile(compute, 95.0);

  s.avgPipeline = mean(pipeline);
  s.p95Pipeline = percentile(pipeline, 95.0);

  s.processingFps = s.avgCompute > 0.0 ? 1000.0 / s.avgCompute : 0.0;

  s.pipelineFps = s.avgPipeline > 0.0 ? 1000.0 / s.avgPipeline : 0.0;

  s.avgCpuUsage = mean(results.cpuUsageSamples);

  if (!results.gpuUsageSamples.empty()) {
    s.avgGpuUsage = mean(results.gpuUsageSamples);
  }

  for (const FrameMetrics &f : results.frames) {
    s.totalFaces += f.faces;
    s.totalEyes += f.eyes;
  }

  return s;
}

void writeSummary(ofstream &log, const PassResults &results, const SummaryNumbers &s) {
  log << "\n================ SUMMARY: " << results.label << " ================\n";

  log << fixed << setprecision(3);

  log << "Frames read                         : " << results.totalFramesRead << "\n";

  log << "Warm-up frames excluded             : " << WARMUP_FRAMES << "\n";

  log << "Measured frames                     : " << results.measuredFrames << "\n";

  log << "Pass wall time                      : " << results.wallTimeMs / 1000.0 << " s\n";

  log << "\nPREPROCESSING\n";
  log << "Average preprocessing               : " << s.avgPreprocess << " ms/frame\n";
  log << "Median preprocessing                : " << s.medianPreprocess << " ms/frame\n";
  log << "P95 preprocessing                   : " << s.p95Preprocess << " ms/frame\n";

  if (results.mode == BenchmarkMode::GPU_ACCELERATED) {
    log << "Average GPU kernel only             : " << s.avgGpuKernel << " ms/frame\n";
    log << "GPU preprocessing above is end-to-end host->GPU + kernel + GPU->host.\n";
  }

  log << "\nDETECTION / ANNOTATION\n";
  log << "Average face detection              : " << s.avgFace << " ms/frame\n";
  log << "Average eye detection               : " << s.avgEye << " ms/frame\n";
  log << "Average ear estimation              : " << s.avgEar << " ms/frame\n";
  log << "Average total detection             : " << s.avgDetection << " ms/frame\n";
  log << "Average drawing                     : " << s.avgDraw << " ms/frame\n";

  log << "\nCORE COMPUTE (preprocess + detection; excludes drawing)\n";
  log << "Average compute                     : " << s.avgCompute << " ms/frame\n";
  log << "Median compute                      : " << s.medianCompute << " ms/frame\n";
  log << "P95 compute                         : " << s.p95Compute << " ms/frame\n";
  log << "Processing throughput               : " << s.processingFps << " FPS\n";

  log << "\nFULL MEASURED PIPELINE (compute + drawing)\n";
  log << "Average pipeline                    : " << s.avgPipeline << " ms/frame\n";
  log << "P95 pipeline                        : " << s.p95Pipeline << " ms/frame\n";
  log << "Pipeline throughput                 : " << s.pipelineFps << " FPS\n";

  log << "\nUTILIZATION\n";
  log << "Average sampled CPU utilization     : " << s.avgCpuUsage << " %\n";

  if (s.avgGpuUsage >= 0.0) {
    log << "Average sampled GPU utilization     : " << s.avgGpuUsage << " %\n";
  } else {
    log << "Average sampled GPU utilization     : N/A\n";
  }

  log << "\nDETECTION COUNTS\n";
  log << "Total face detections               : " << s.totalFaces << "\n";
  log << "Total eye detections                : " << s.totalEyes << "\n";
}

void writeComparison(ofstream &log, const SummaryNumbers &cpu, const SummaryNumbers &gpu) {
  log << "\n================ CPU VS GPU COMPARISON ================\n";
  log << fixed << setprecision(3);

  if (gpu.avgPreprocess > 0.0) {
    log << "Preprocessing speedup (CPU/GPU E2E) : " << cpu.avgPreprocess / gpu.avgPreprocess << "x\n";
  }

  if (gpu.avgCompute > 0.0) {
    log << "Core compute speedup                 : " << cpu.avgCompute / gpu.avgCompute << "x\n";
  }

  if (gpu.avgPipeline > 0.0) {
    log << "Full measured pipeline speedup       : " << cpu.avgPipeline / gpu.avgPipeline << "x\n";
  }

  log << "CPU processing throughput            : " << cpu.processingFps << " FPS\n";

  log << "GPU-accelerated processing throughput: " << gpu.processingFps << " FPS\n";

  log << "CPU P95 compute                      : " << cpu.p95Compute << " ms/frame\n";

  log << "GPU P95 compute                      : " << gpu.p95Compute << " ms/frame\n";

  log << "CPU total face detections            : " << cpu.totalFaces << "\n";

  log << "GPU total face detections            : " << gpu.totalFaces << "\n";

  log << "CPU total eye detections             : " << cpu.totalEyes << "\n";

  log << "GPU total eye detections             : " << gpu.totalEyes << "\n";

  log << "\nIMPORTANT: Haar face/eye detection runs on the CPU in BOTH passes.\n";
  log << "The controlled variable is CPU preprocessing versus OpenCL GPU preprocessing.\n";
  log << "Video decoding, recording, and log-file I/O are outside the per-frame compute timers.\n";
}

int main(int argc, char *argv[]) {

  if (argc > 1) {
    RECORD_SECONDS = std::stoi(argv[1]);
  }

  cout << "CPU vs GPU Controlled Video Test Bench\n";
  cout << "======================================\n";
  cout << "Record duration " << RECORD_SECONDS << "s" << endl;
  cout << "1) Record one camera video " << PRIMARY_INPUT_VIDEO << endl;
  cout << "2) Replay it through CPU preprocessing + CPU Haar detection" << endl;
  cout << "3) Replay the same video through GPU preprocessing + CPU Haar detection" << endl;
  cout << "4) Write per-frame data and final comparison to " << BENCHMARK_LOG << endl;

  CascadeClassifier faceCascade;
  CascadeClassifier eyeCascade;

  string cascadePath = "/usr/share/opencv4/haarcascades/";

  if (!faceCascade.load(cascadePath + "haarcascade_frontalface_default.xml") || !eyeCascade.load(cascadePath + "haarcascade_eye.xml")) {
    cerr << "Could not load Haar cascade classifiers";
    return 1;
  }

  // Phase 1: record exactly one test video.
  RecordingInfo recording;

  if (!recordTestVideo(recording)) {
    return 1;
  }

  // cout << "\nCooling down for " << COOLDOWN_SECONDS << " seconds before CPU benchmark..." << endl;
  // this_thread::sleep_for(seconds(COOLDOWN_SECONDS));

  // Phase 2: CPU benchmark.
  PassResults cpuResults;

  if (!runBenchmarkPass(recording.path, BenchmarkMode::CPU, faceCascade, eyeCascade, nullptr, cpuResults)) {
    cerr << "CPU benchmark failed\n";
    return 1;
  }

  // cout << "\nCooling down for " << COOLDOWN_SECONDS << " seconds before GPU benchmark..." << endl;
  // this_thread::sleep_for(seconds(COOLDOWN_SECONDS));

  // Initialize OpenCL only after recording and CPU benchmark.
  OpenCLContext ocl;
  if (!initOpenCL(ocl, "kernel.cl", recording.width, recording.height)) {
    cerr << "Failed to initialize OpenCL GPU pipeline" << endl;
    return 1;
  }

  // Phase 3: GPU-accelerated benchmark over same video.
  PassResults gpuResults;

  bool gpuOk = runBenchmarkPass(recording.path, BenchmarkMode::GPU_ACCELERATED, faceCascade, eyeCascade, &ocl, gpuResults);

  cleanupOpenCL(ocl);

  if (!gpuOk) {
    cerr << "GPU benchmark failed\n";
    return 1;
  }

  // Phase 4: write all results after timing has finished.
  ofstream benchmarkLog(BENCHMARK_LOG, ios::out | ios::trunc);

  if (!benchmarkLog.is_open()) {
    cerr << "Could not create " << BENCHMARK_LOG << "\n";
    return 1;
  }

  benchmarkLog << "================ CONTROLLED CPU VS GPU VIDEO TEST BENCH ================\n";
  benchmarkLog << "Recorded input video                : " << recording.path << "\n";
  benchmarkLog << "Resolution                          : " << recording.width << "x" << recording.height << "\n";
  benchmarkLog << "Requested camera rate               : " << recording.fps << " FPS\n";
  benchmarkLog << "Recorded frames                     : " << recording.frames << "\n";
  benchmarkLog << "Recorded duration                   : " << fixed << setprecision(3) << recording.durationSec << " s\n";
  benchmarkLog << "Warm-up frames excluded per pass    : " << WARMUP_FRAMES << "\n";
  benchmarkLog << "\nCPU PASS: CPU cvtColor + resize + CPU Haar face/eye detection.\n";
  benchmarkLog << "GPU PASS: OpenCL BGR->gray/downscale + CPU Haar face/eye detection.\n";
  benchmarkLog << "Both passes read the exact same recorded video file.\n";
  benchmarkLog << "Decode time is not included in per-frame compute metrics.\n";

  writeFrameTable(benchmarkLog, cpuResults);
  SummaryNumbers cpuSummary = summarize(cpuResults);
  writeSummary(benchmarkLog, cpuResults, cpuSummary);

  writeFrameTable(benchmarkLog, gpuResults);
  SummaryNumbers gpuSummary = summarize(gpuResults);
  writeSummary(benchmarkLog, gpuResults, gpuSummary);

  writeComparison(benchmarkLog, cpuSummary, gpuSummary);

  benchmarkLog.close();

  // console output
  cout << "\n================ TEST COMPLETE ================\n";
  cout << fixed << setprecision(3);
  cout << "CPU preprocess       : " << cpuSummary.avgPreprocess << " ms/frame\n";
  cout << "GPU preprocess E2E   : " << gpuSummary.avgPreprocess << " ms/frame\n";
  cout << "GPU kernel only      : " << gpuSummary.avgGpuKernel << " ms/frame\n";
  cout << "CPU processing       : " << cpuSummary.processingFps << " FPS\n";
  cout << "GPU-accelerated      : " << gpuSummary.processingFps << " FPS\n";

  if (gpuSummary.avgPreprocess > 0.0) {
    cout << "Preprocess speedup    : " << cpuSummary.avgPreprocess / gpuSummary.avgPreprocess << "x\n";
  }

  if (gpuSummary.avgCompute > 0.0) {
    cout << "Core compute speedup  : " << cpuSummary.avgCompute / gpuSummary.avgCompute << "x\n";
  }

  cout << "Results saved to      : " << BENCHMARK_LOG << "\n";
  cout << "Recorded test video   : " << recording.path << "\n";

  return 0;
}
