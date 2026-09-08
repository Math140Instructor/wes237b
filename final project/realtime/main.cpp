#include <CL/cl.h>
#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <netinet/in.h>
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

// ============================================================
// Global running flag
// ============================================================

atomic<bool> running(true);

// ============================================================
// Terminal state
// ============================================================

termios originalTerminalSettings;
int originalTerminalFlags = 0;
bool terminalConfigured = false;

// ============================================================
// OpenCL Context
// ============================================================

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

// ============================================================
// CPU usage snapshot
// ============================================================

struct CpuSnapshot {
  unsigned long long totalUser = 0;
  unsigned long long totalUserLow = 0;
  unsigned long long totalSys = 0;
  unsigned long long totalIdle = 0;
};

// ============================================================
// Signal handler
// ============================================================

void signalHandler(int) {
  running = false;
}

// ============================================================
// Terminal keyboard handling
// ============================================================

bool enableNonBlockingInput() {

  if (!isatty(STDIN_FILENO)) {
    cerr << "Warning: stdin is not a terminal. Hotkeys disabled.\n";
    return false;
  }

  if (tcgetattr(STDIN_FILENO, &originalTerminalSettings) != 0) {
    cerr << "Warning: could not read terminal settings.\n";
    return false;
  }

  termios terminalSettings = originalTerminalSettings;

  terminalSettings.c_lflag &= ~(ICANON | ECHO);

  if (tcsetattr(STDIN_FILENO, TCSANOW, &terminalSettings) != 0) {
    cerr << "Warning: could not configure terminal.\n";
    return false;
  }

  originalTerminalFlags =
      fcntl(STDIN_FILENO, F_GETFL, 0);

  if (originalTerminalFlags < 0) {
    originalTerminalFlags = 0;
  }

  fcntl(
      STDIN_FILENO,
      F_SETFL,
      originalTerminalFlags | O_NONBLOCK);

  terminalConfigured = true;

  return true;
}

void restoreTerminal() {

  if (!terminalConfigured) {
    return;
  }

  tcsetattr(
      STDIN_FILENO,
      TCSANOW,
      &originalTerminalSettings);

  fcntl(
      STDIN_FILENO,
      F_SETFL,
      originalTerminalFlags);

  terminalConfigured = false;
}

// ============================================================
// CPU usage
// ============================================================

CpuSnapshot readCpuSnapshot() {

  CpuSnapshot snap;

  ifstream statFile("/proc/stat");

  string line;

  if (getline(statFile, line)) {

    if (line.substr(0, 3) == "cpu") {

      stringstream ss(line);

      string cpuLabel;

      ss >> cpuLabel
         >> snap.totalUser
         >> snap.totalUserLow
         >> snap.totalSys
         >> snap.totalIdle;
    }
  }

  return snap;
}

double calculateCpuUsage(
    const CpuSnapshot &prev,
    const CpuSnapshot &curr) {

  unsigned long long prevTotal =
      prev.totalUser +
      prev.totalUserLow +
      prev.totalSys +
      prev.totalIdle;

  unsigned long long currTotal =
      curr.totalUser +
      curr.totalUserLow +
      curr.totalSys +
      curr.totalIdle;

  unsigned long long totalDelta =
      currTotal - prevTotal;

  unsigned long long idleDelta =
      curr.totalIdle - prev.totalIdle;

  if (totalDelta == 0) {
    return 0.0;
  }

  return
      (1.0 -
       static_cast<double>(idleDelta) /
           static_cast<double>(totalDelta)) *
      100.0;
}

// ============================================================
// GPU usage
// ============================================================

double readGpuUsage() {

  ifstream gpubusyFile(
      "/sys/class/kgsl/kgsl-3d0/gpubusy");

  if (gpubusyFile.is_open()) {

    unsigned long long busyCycles = 0;
    unsigned long long totalCycles = 0;

    gpubusyFile >>
        busyCycles >>
        totalCycles;

    if (totalCycles > 0) {

      return
          static_cast<double>(busyCycles) /
          static_cast<double>(totalCycles) *
          100.0;
    }
  }

  ifstream altGpuFile(
      "/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage");

  if (altGpuFile.is_open()) {

    double pct = 0.0;

    altGpuFile >> pct;

    return pct;
  }

  return -1.0;
}

// ============================================================
// Initialize OpenCL
// ============================================================

bool initOpenCL(
    OpenCLContext &ocl,
    const string &kernelFile,
    int width,
    int height) {

  cl_int err;

  cl_uint numPlatforms = 0;

  err =
      clGetPlatformIDs(
          1,
          &ocl.platform,
          &numPlatforms);

  if (err != CL_SUCCESS ||
      numPlatforms == 0) {

    cerr << "Could not find OpenCL platform\n";

    return false;
  }

  err =
      clGetDeviceIDs(
          ocl.platform,
          CL_DEVICE_TYPE_GPU,
          1,
          &ocl.device,
          nullptr);

  if (err != CL_SUCCESS) {

    cerr << "Could not find GPU OpenCL device\n";

    return false;
  }

  ocl.context =
      clCreateContext(
          nullptr,
          1,
          &ocl.device,
          nullptr,
          nullptr,
          &err);

  if (err != CL_SUCCESS) {

    cerr << "Could not create OpenCL context\n";

    return false;
  }

  ocl.queue =
      clCreateCommandQueue(
          ocl.context,
          ocl.device,
          CL_QUEUE_PROFILING_ENABLE,
          &err);

  if (err != CL_SUCCESS) {

    cerr << "Could not create OpenCL queue\n";

    return false;
  }

  ifstream file(kernelFile);

  if (!file.is_open()) {

    cerr << "Could not open "
         << kernelFile
         << "\n";

    return false;
  }

  string srcStr(
      (istreambuf_iterator<char>(file)),
      istreambuf_iterator<char>());

  const char *src =
      srcStr.c_str();

  size_t length =
      srcStr.length();

  ocl.program =
      clCreateProgramWithSource(
          ocl.context,
          1,
          &src,
          &length,
          &err);

  if (err != CL_SUCCESS) {

    cerr << "Could not create OpenCL program\n";

    return false;
  }

  err =
      clBuildProgram(
          ocl.program,
          1,
          &ocl.device,
          "-cl-fast-relaxed-math",
          nullptr,
          nullptr);

  if (err != CL_SUCCESS) {

    size_t logSize = 0;

    clGetProgramBuildInfo(
        ocl.program,
        ocl.device,
        CL_PROGRAM_BUILD_LOG,
        0,
        nullptr,
        &logSize);

    vector<char> log(logSize);

    clGetProgramBuildInfo(
        ocl.program,
        ocl.device,
        CL_PROGRAM_BUILD_LOG,
        logSize,
        log.data(),
        nullptr);

    cerr << "OpenCL Build Error:\n"
         << log.data()
         << "\n";

    return false;
  }

  ocl.kernel =
      clCreateKernel(
          ocl.program,
          "bgr_to_gray_downscale_2x",
          &err);

  if (err != CL_SUCCESS) {

    cerr << "Could not create OpenCL kernel\n";

    return false;
  }

  size_t srcBytes =
      static_cast<size_t>(width) *
      static_cast<size_t>(height) *
      3;

  size_t dstBytes =
      static_cast<size_t>(width / 2) *
      static_cast<size_t>(height / 2);

  ocl.d_src =
      clCreateBuffer(
          ocl.context,
          CL_MEM_READ_ONLY,
          srcBytes,
          nullptr,
          &err);

  if (err != CL_SUCCESS) {

    cerr << "Could not create OpenCL source buffer\n";

    return false;
  }

  ocl.d_dst =
      clCreateBuffer(
          ocl.context,
          CL_MEM_WRITE_ONLY,
          dstBytes,
          nullptr,
          &err);

  if (err != CL_SUCCESS) {

    cerr << "Could not create OpenCL destination buffer\n";

    return false;
  }

  return true;
}

// ============================================================
// Cleanup OpenCL
// ============================================================

void cleanupOpenCL(
    OpenCLContext &ocl) {

  if (ocl.d_src)
    clReleaseMemObject(ocl.d_src);

  if (ocl.d_dst)
    clReleaseMemObject(ocl.d_dst);

  if (ocl.kernel)
    clReleaseKernel(ocl.kernel);

  if (ocl.program)
    clReleaseProgram(ocl.program);

  if (ocl.queue)
    clReleaseCommandQueue(ocl.queue);

  if (ocl.context)
    clReleaseContext(ocl.context);
}

// ============================================================
// CPU grayscale + 2x downscale
// ============================================================

double processCpu(
    const Mat &frame,
    Mat &gray,
    Mat &smallGray,
    int smallWidth,
    int smallHeight) {

  auto start =
      high_resolution_clock::now();

  cvtColor(
      frame,
      gray,
      COLOR_BGR2GRAY);

  resize(
      gray,
      smallGray,
      Size(
          smallWidth,
          smallHeight),
      0,
      0,
      INTER_LINEAR);

  auto end =
      high_resolution_clock::now();

  return
      duration_cast<microseconds>(
          end - start)
          .count() /
      1000.0;
}

// ============================================================
// GPU grayscale + 2x downscale
// ============================================================

bool processGpu(
    OpenCLContext &ocl,
    const Mat &frame,
    Mat &smallGray,
    int fullWidth,
    int fullHeight,
    int smallWidth,
    int smallHeight,
    double &kernelMs) {

  cl_int err;

  size_t sourceBytes =
      static_cast<size_t>(fullHeight) *
      frame.step;

  err =
      clEnqueueWriteBuffer(
          ocl.queue,
          ocl.d_src,
          CL_FALSE,
          0,
          sourceBytes,
          frame.data,
          0,
          nullptr,
          nullptr);

  if (err != CL_SUCCESS) {
    return false;
  }

  int srcStep =
      static_cast<int>(frame.step);

  int dstStep =
      static_cast<int>(smallGray.step);

  err |=
      clSetKernelArg(
          ocl.kernel,
          0,
          sizeof(cl_mem),
          &ocl.d_src);

  err |=
      clSetKernelArg(
          ocl.kernel,
          1,
          sizeof(cl_mem),
          &ocl.d_dst);

  err |=
      clSetKernelArg(
          ocl.kernel,
          2,
          sizeof(int),
          &fullWidth);

  err |=
      clSetKernelArg(
          ocl.kernel,
          3,
          sizeof(int),
          &fullHeight);

  err |=
      clSetKernelArg(
          ocl.kernel,
          4,
          sizeof(int),
          &srcStep);

  err |=
      clSetKernelArg(
          ocl.kernel,
          5,
          sizeof(int),
          &dstStep);

  if (err != CL_SUCCESS) {
    return false;
  }

  size_t globalWorkSize[2] = {
      static_cast<size_t>(smallWidth),
      static_cast<size_t>(smallHeight)};

  size_t localWorkSize[2] = {
      16,
      16};

  cl_event kernelEvent = nullptr;

  err =
      clEnqueueNDRangeKernel(
          ocl.queue,
          ocl.kernel,
          2,
          nullptr,
          globalWorkSize,
          localWorkSize,
          0,
          nullptr,
          &kernelEvent);

  if (err != CL_SUCCESS) {
    return false;
  }

  err =
      clEnqueueReadBuffer(
          ocl.queue,
          ocl.d_dst,
          CL_TRUE,
          0,
          static_cast<size_t>(smallWidth) *
              static_cast<size_t>(smallHeight),
          smallGray.data,
          0,
          nullptr,
          nullptr);

  if (err != CL_SUCCESS) {

    clReleaseEvent(kernelEvent);

    return false;
  }

  clWaitForEvents(
      1,
      &kernelEvent);

  cl_ulong kernelStart = 0;
  cl_ulong kernelEnd = 0;

  clGetEventProfilingInfo(
      kernelEvent,
      CL_PROFILING_COMMAND_START,
      sizeof(kernelStart),
      &kernelStart,
      nullptr);

  clGetEventProfilingInfo(
      kernelEvent,
      CL_PROFILING_COMMAND_END,
      sizeof(kernelEnd),
      &kernelEnd,
      nullptr);

  kernelMs =
      static_cast<double>(
          kernelEnd - kernelStart) *
      1e-6;

  clReleaseEvent(kernelEvent);

  return true;
}

// ============================================================
// Draw benchmark HUD
// ============================================================

void drawBenchmarkHUD(
    Mat &img,
    bool useGpu,
    double cpuUsage,
    double gpuUsage,
    double processingMs,
    double currentFps,
    int frameIdx) {

  vector<string> lines;

  stringstream ssMode;
  stringstream ssCpu;
  stringstream ssGpu;
  stringstream ssProcessing;
  stringstream ssFps;

  ssMode
      << "MODE: "
      << (useGpu ? "GPU" : "CPU");

  lines.push_back(
      ssMode.str());

  ssCpu
      << fixed
      << setprecision(1)
      << "CPU: "
      << cpuUsage
      << " %";

  lines.push_back(
      ssCpu.str());

  if (gpuUsage >= 0.0) {

    ssGpu
        << fixed
        << setprecision(1)
        << "GPU: "
        << gpuUsage
        << " %";

  } else {

    ssGpu << "GPU: N/A";
  }

  lines.push_back(
      ssGpu.str());

  if (useGpu) {

    ssProcessing
        << fixed
        << setprecision(2)
        << "GPU Kernel: "
        << processingMs
        << " ms";

  } else {

    ssProcessing
        << fixed
        << setprecision(2)
        << "CPU Gray+Resize: "
        << processingMs
        << " ms";
  }

  lines.push_back(
      ssProcessing.str());

  ssFps
      << fixed
      << setprecision(1)
      << "FPS: "
      << currentFps
      << " (F:"
      << frameIdx
      << ")";

  lines.push_back(
      ssFps.str());

  int fontFace =
      FONT_HERSHEY_SIMPLEX;

  double fontScale =
      0.5;

  int thickness =
      1;

  int lineSpacing =
      22;

  int margin =
      12;

  int maxTextWidth =
      0;

  for (const auto &text : lines) {

    int baseline =
        0;

    Size textSize =
        getTextSize(
            text,
            fontFace,
            fontScale,
            thickness,
            &baseline);

    maxTextWidth =
        max(
            maxTextWidth,
            textSize.width);
  }

  int boxWidth =
      maxTextWidth +
      margin * 2;

  int boxHeight =
      static_cast<int>(
          lines.size()) *
          lineSpacing +
      margin;

  int boxX =
      img.cols -
      boxWidth -
      10;

  int boxY =
      10;

  Rect hudRect(
      boxX,
      boxY,
      boxWidth,
      boxHeight);

  if (hudRect.x >= 0 &&
      hudRect.y >= 0 &&
      hudRect.x + hudRect.width <= img.cols &&
      hudRect.y + hudRect.height <= img.rows) {

    Mat roi =
        img(hudRect);

    Mat overlay;

    roi.copyTo(
        overlay);

    rectangle(
        overlay,
        Rect(
            0,
            0,
            boxWidth,
            boxHeight),
        Scalar(
            15,
            15,
            15),
        FILLED);

    addWeighted(
        overlay,
        0.7,
        roi,
        0.3,
        0,
        roi);

    rectangle(
        img,
        hudRect,
        Scalar(
            80,
            80,
            80),
        1);
  }

  int textY =
      boxY +
      margin +
      12;

  for (const auto &text : lines) {

    putText(
        img,
        text,
        Point(
            boxX + margin,
            textY),
        fontFace,
        fontScale,
        Scalar(
            0,
            255,
            255),
        thickness,
        LINE_AA);

    textY +=
        lineSpacing;
  }
}

// ============================================================
// Send all bytes over socket
// ============================================================

bool sendAll(
    int socketFd,
    const void *data,
    size_t length) {

  const char *buffer =
      static_cast<const char *>(
          data);

  while (length > 0) {

    ssize_t sent =
        send(
            socketFd,
            buffer,
            length,
            MSG_NOSIGNAL);

    if (sent <= 0) {
      return false;
    }

    buffer += sent;

    length -=
        static_cast<size_t>(sent);
  }

  return true;
}

// ============================================================
// Main
// ============================================================

int main(
    int argc,
    char *argv[]) {

  signal(
      SIGINT,
      signalHandler);

  signal(
      SIGTERM,
      signalHandler);

  // ==========================================================
  // CLI server settings
  // ==========================================================

  string serverIp =
      "localhost";

  int port =
      5000;

  if (argc == 3) {

    serverIp =
        argv[1];

    try {

      port =
          stoi(argv[2]);

    } catch (...) {

      cerr
          << "Invalid port: "
          << argv[2]
          << "\n";

      return 1;
    }

  } else if (argc != 1) {

    cerr
        << "Usage: "
        << argv[0]
        << " [IP port]\n";

    cerr
        << "Default: localhost 5000\n";

    cerr
        << "Example: "
        << argv[0]
        << " 0.0.0.0 5000\n";

    return 1;
  }

  if (port < 1 ||
      port > 65535) {

    cerr
        << "Port must be between 1 and 65535\n";

    return 1;
  }

  // ==========================================================
  // Keyboard
  // ==========================================================

  bool keyboardEnabled =
      enableNonBlockingInput();

  atexit(
      restoreTerminal);

  bool useGpu =
      false;

  cout
      << "\nStarting in CPU MODE\n";

  if (keyboardEnabled) {

    cout
        << "Press G to toggle CPU/GPU\n";

    cout
        << "Press Q to quit\n";
  }

  // ==========================================================
  // Camera
  // ==========================================================

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

    cerr
        << "Could not open RB3 camera\n";

    return 1;
  }

  cout
      << "RB3 camera opened\n";

  // ==========================================================
  // Haar cascades
  // ==========================================================

  CascadeClassifier faceCascade;
  CascadeClassifier eyeCascade;

  string cascadePath =
      "/usr/share/opencv4/haarcascades/";

  if (!faceCascade.load(
          cascadePath +
          "haarcascade_frontalface_default.xml") ||
      !eyeCascade.load(
          cascadePath +
          "haarcascade_eye.xml")) {

    cerr
        << "Could not load cascade classifiers\n";

    return 1;
  }

  // ==========================================================
  // First camera frame
  // ==========================================================

  Mat frame;

  if (!cap.read(frame) ||
      frame.empty()) {

    cerr
        << "Could not read first camera frame\n";

    return 1;
  }

  int fullWidth =
      frame.cols;

  int fullHeight =
      frame.rows;

  int smallWidth =
      fullWidth / 2;

  int smallHeight =
      fullHeight / 2;

  cout
      << "Camera resolution: "
      << fullWidth
      << "x"
      << fullHeight
      << "\n";

  // ==========================================================
  // OpenCL
  // ==========================================================

  OpenCLContext ocl;

  if (!initOpenCL(
          ocl,
          "kernel.cl",
          fullWidth,
          fullHeight)) {

    cerr
        << "Failed to initialize OpenCL GPU pipeline\n";

    return 1;
  }

  cout
      << "OpenCL GPU pipeline initialized\n";

  // ==========================================================
  // HTTP server
  // ==========================================================

  int serverSocket =
      socket(
          AF_INET,
          SOCK_STREAM,
          0);

  if (serverSocket < 0) {

    cerr
        << "Could not create server socket\n";

    cleanupOpenCL(
        ocl);

    return 1;
  }

  int reuse =
      1;

  setsockopt(
      serverSocket,
      SOL_SOCKET,
      SO_REUSEADDR,
      &reuse,
      sizeof(reuse));

  sockaddr_in serverAddress{};

  serverAddress.sin_family =
      AF_INET;

  serverAddress.sin_port =
      htons(port);

  if (serverIp == "localhost") {

    serverAddress.sin_addr.s_addr =
        htonl(
            INADDR_LOOPBACK);

  } else if (serverIp == "0.0.0.0") {

    serverAddress.sin_addr.s_addr =
        htonl(
            INADDR_ANY);

  } else {

    if (inet_pton(
            AF_INET,
            serverIp.c_str(),
            &serverAddress.sin_addr) != 1) {

      cerr
          << "Invalid server address: "
          << serverIp
          << "\n";

      close(
          serverSocket);

      cleanupOpenCL(
          ocl);

      return 1;
    }
  }

  // Retry bind indefinitely
  while (
      running &&
      bind(
          serverSocket,
          reinterpret_cast<sockaddr *>(
              &serverAddress),
          sizeof(serverAddress)) < 0) {

    cerr
        << "Could not bind to "
        << serverIp
        << ":"
        << port
        << ". Retrying...\n";

    sleep(1);
  }

  if (!running) {

    close(
        serverSocket);

    cleanupOpenCL(
        ocl);

    return 0;
  }

  if (listen(
          serverSocket,
          1) < 0) {

    cerr
        << "Could not listen on port "
        << port
        << "\n";

    close(
        serverSocket);

    cleanupOpenCL(
        ocl);

    return 1;
  }

  cout
      << "\nVideo available at:\n";

  cout
      << "http://"
      << serverIp
      << ":"
      << port
      << "\n";

  cout
      << "Camera will continue running whether or not a viewer is connected.\n";

  // Make accept() non-blocking so the camera loop never waits for a viewer.
  int serverFlags =
      fcntl(
          serverSocket,
          F_GETFL,
          0);

  if (serverFlags < 0 ||
      fcntl(
          serverSocket,
          F_SETFL,
          serverFlags | O_NONBLOCK) < 0) {

    perror("fcntl server socket");

    close(
        serverSocket);

    cleanupOpenCL(
        ocl);

    return 1;
  }

  int clientSocket = -1;

  // ==========================================================
  // HTTP MJPEG response
  // ==========================================================

  const string httpHeader =
      "HTTP/1.1 200 OK\r\n"
      "Cache-Control: no-cache, no-store, must-revalidate\r\n"
      "Pragma: no-cache\r\n"
      "Expires: 0\r\n"
      "Connection: close\r\n"
      "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
      "\r\n";

  // ==========================================================
  // Runtime variables
  // ==========================================================

  Mat gray;

  Mat smallGray(
      smallHeight,
      smallWidth,
      CV_8UC1);

  const double invScale =
      2.0;

  CpuSnapshot lastCpuSnap =
      readCpuSnapshot();

  auto lastBenchmarkTime =
      high_resolution_clock::now();

  auto lastFrameTimestamp =
      high_resolution_clock::now();

  auto startTime =
      high_resolution_clock::now();

  double currentCpuPct =
      0.0;

  double currentGpuPct =
      0.0;

  double currentProcessingMs =
      0.0;

  double totalCpuProcessingMs =
      0.0;

  double totalGpuKernelMs =
      0.0;

  int cpuFrameCount =
      0;

  int gpuFrameCount =
      0;

  int frameCount =
      0;

  cout
      << "\n================ LIVE BENCHMARK ================\n";

  cout
      << "MODE: CPU\n";

  // ==========================================================
  // LIVE CAMERA + PROCESSING + STREAM LOOP
  // ==========================================================

  while (running) {

    // --------------------------------------------------------
    // Keyboard input
    // --------------------------------------------------------

    if (keyboardEnabled) {

      char key;

      while (
          read(
              STDIN_FILENO,
              &key,
              1) > 0) {

        if (key == 'g' ||
            key == 'G') {

          useGpu =
              !useGpu;

          cout
              << "\n====================================\n";

          cout
              << "MODE CHANGED TO: "
              << (useGpu ? "GPU" : "CPU")
              << "\n";

          cout
              << "====================================\n";

        } else if (
            key == 'q' ||
            key == 'Q') {

          running =
              false;

          break;
        }
      }
    }

    if (!running) {
      break;
    }

    // --------------------------------------------------------
    // Capture frame
    // --------------------------------------------------------

    if (!cap.read(frame) ||
        frame.empty()) {

      cerr
          << "Frame capture error\n";

      break;
    }

    // --------------------------------------------------------
    // CPU/GPU selectable preprocessing
    // --------------------------------------------------------

    if (useGpu) {

      double kernelMs =
          0.0;

      if (!processGpu(
              ocl,
              frame,
              smallGray,
              fullWidth,
              fullHeight,
              smallWidth,
              smallHeight,
              kernelMs)) {

        cerr
            << "GPU processing failed\n";

        break;
      }

      currentProcessingMs =
          kernelMs;

      totalGpuKernelMs +=
          kernelMs;

      gpuFrameCount++;

    } else {

      currentProcessingMs =
          processCpu(
              frame,
              gray,
              smallGray,
              smallWidth,
              smallHeight);

      totalCpuProcessingMs +=
          currentProcessingMs;

      cpuFrameCount++;
    }

    // --------------------------------------------------------
    // Face detection
    // --------------------------------------------------------

    vector<Rect> faces;

    faceCascade.detectMultiScale(
        smallGray,
        faces,
        1.1,
        4,
        0,
        Size(
            30,
            30));

    for (const Rect &smallFace : faces) {

      Rect face(
          cvRound(
              smallFace.x *
              invScale),

          cvRound(
              smallFace.y *
              invScale),

          cvRound(
              smallFace.width *
              invScale),

          cvRound(
              smallFace.height *
              invScale));

      // Clamp face rectangle
      face &=
          Rect(
              0,
              0,
              frame.cols,
              frame.rows);

      if (face.width <= 0 ||
          face.height <= 0) {

        continue;
      }

      rectangle(
          frame,
          face,
          Scalar(
              0,
              0,
              255),
          2);

      putText(
          frame,
          "Face",
          Point(
              face.x,
              max(
                  0,
                  face.y - 5)),
          FONT_HERSHEY_SIMPLEX,
          0.5,
          Scalar(
              0,
              0,
              255),
          1);

      Mat faceROI =
          smallGray(
              smallFace);

      vector<Rect> eyes;

      eyeCascade.detectMultiScale(
          faceROI,
          eyes,
          1.1,
          4,
          0,
          Size(
              15,
              15));

      for (const Rect &smallEye : eyes) {

        Rect eyeGlobal(
            cvRound(
                (smallFace.x +
                 smallEye.x) *
                invScale),

            cvRound(
                (smallFace.y +
                 smallEye.y) *
                invScale),

            cvRound(
                smallEye.width *
                invScale),

            cvRound(
                smallEye.height *
                invScale));

        eyeGlobal &=
            Rect(
                0,
                0,
                frame.cols,
                frame.rows);

        if (eyeGlobal.width <= 0 ||
            eyeGlobal.height <= 0) {

          continue;
        }

        rectangle(
            frame,
            eyeGlobal,
            Scalar(
                0,
                255,
                0),
            2);

        putText(
            frame,
            "Eye",
            Point(
                eyeGlobal.x,
                max(
                    0,
                    eyeGlobal.y - 4)),
            FONT_HERSHEY_SIMPLEX,
            0.4,
            Scalar(
                0,
                255,
                0),
            1);
      }

      // ------------------------------------------------------
      // Approximate ear positions
      // ------------------------------------------------------

      int earWidth =
          static_cast<int>(
              face.width *
              0.18);

      int earHeight =
          static_cast<int>(
              face.height *
              0.35);

      int earY =
          face.y +
          static_cast<int>(
              face.height *
              0.28);

      if (earWidth > 0 &&
          earHeight > 0) {

        Rect leftEarRect(
            max(
                0,
                face.x -
                    static_cast<int>(
                        earWidth *
                        0.6)),
            earY,
            earWidth,
            earHeight);

        Rect rightEarRect(
            min(
                frame.cols -
                    earWidth,
                face.x +
                    face.width -
                    static_cast<int>(
                        earWidth *
                        0.4)),
            earY,
            earWidth,
            earHeight);

        leftEarRect &=
            Rect(
                0,
                0,
                frame.cols,
                frame.rows);

        rightEarRect &=
            Rect(
                0,
                0,
                frame.cols,
                frame.rows);

        if (leftEarRect.width > 0 &&
            leftEarRect.height > 0) {

          rectangle(
              frame,
              leftEarRect,
              Scalar(
                  255,
                  0,
                  0),
              2);

          putText(
              frame,
              "L Ear",
              Point(
                  leftEarRect.x,
                  max(
                      0,
                      leftEarRect.y - 4)),
              FONT_HERSHEY_SIMPLEX,
              0.4,
              Scalar(
                  255,
                  0,
                  0),
              1);
        }

        if (rightEarRect.width > 0 &&
            rightEarRect.height > 0) {

          rectangle(
              frame,
              rightEarRect,
              Scalar(
                  255,
                  255,
                  0),
              2);

          putText(
              frame,
              "R Ear",
              Point(
                  rightEarRect.x,
                  max(
                      0,
                      rightEarRect.y - 4)),
              FONT_HERSHEY_SIMPLEX,
              0.4,
              Scalar(
                  255,
                  255,
                  0),
              1);
        }
      }

      // Only first detected face
      break;
    }

    // --------------------------------------------------------
    // FPS
    // --------------------------------------------------------

    auto now =
        high_resolution_clock::now();

    double frameDeltaMs =
        duration_cast<microseconds>(
            now -
            lastFrameTimestamp)
            .count() /
        1000.0;

    lastFrameTimestamp =
        now;

    double instantFps =
        frameDeltaMs > 0.0
            ? 1000.0 /
                  frameDeltaMs
            : 0.0;

    // --------------------------------------------------------
    // CPU/GPU usage every 500 ms
    // --------------------------------------------------------

    double intervalSec =
        duration_cast<milliseconds>(
            now -
            lastBenchmarkTime)
            .count() /
        1000.0;

    if (intervalSec >= 0.5) {

      CpuSnapshot currentCpuSnap =
          readCpuSnapshot();

      currentCpuPct =
          calculateCpuUsage(
              lastCpuSnap,
              currentCpuSnap);

      currentGpuPct =
          readGpuUsage();

      lastCpuSnap =
          currentCpuSnap;

      lastBenchmarkTime =
          now;

      cout
          << fixed
          << setprecision(1);

      cout
          << "[Benchmark] "
          << "MODE: "
          << (useGpu ? "GPU" : "CPU")
          << " | Frames: "
          << frameCount
          << " | CPU: "
          << currentCpuPct
          << "% | ";

      if (currentGpuPct >= 0.0) {

        cout
            << "GPU: "
            << currentGpuPct
            << "% | ";

      } else {

        cout
            << "GPU: N/A | ";
      }

      cout
          << (useGpu
                  ? "GPU Kernel: "
                  : "CPU Gray+Resize: ")
          << setprecision(2)
          << currentProcessingMs
          << " ms | FPS: "
          << setprecision(1)
          << instantFps
          << "\n";
    }

    // --------------------------------------------------------
    // HUD
    // --------------------------------------------------------

    drawBenchmarkHUD(
        frame,
        useGpu,
        currentCpuPct,
        currentGpuPct,
        currentProcessingMs,
        instantFps,
        frameCount + 1);

    // --------------------------------------------------------
    // Accept a viewer without ever blocking the camera loop.
    // --------------------------------------------------------

    if (clientSocket < 0) {

      sockaddr_in clientAddress{};

      socklen_t clientLength =
          sizeof(clientAddress);

      int newClientSocket =
          accept(
              serverSocket,
              reinterpret_cast<sockaddr *>(
                  &clientAddress),
              &clientLength);

      if (newClientSocket >= 0) {

        // Never allow a slow viewer to stall camera processing.
        int clientFlags =
            fcntl(
                newClientSocket,
                F_GETFL,
                0);

        if (clientFlags >= 0) {
          fcntl(
              newClientSocket,
              F_SETFL,
              clientFlags | O_NONBLOCK);
        }

        if (!sendAll(
                newClientSocket,
                httpHeader.data(),
                httpHeader.size())) {

          close(
              newClientSocket);

        } else {

          clientSocket =
              newClientSocket;

          cout
              << "Viewer connected\n";
        }

      } else if (errno != EAGAIN &&
                 errno != EWOULDBLOCK &&
                 errno != EINTR) {

        perror("accept");
      }
    }

    // No viewer: skip JPEG/network work and immediately process
    // the next camera frame.
    if (clientSocket < 0) {
      frameCount++;
      continue;
    }

    // --------------------------------------------------------
    // JPEG encode only while a viewer is connected.
    // --------------------------------------------------------

    vector<uchar> jpeg;

    vector<int> jpegParams = {
        IMWRITE_JPEG_QUALITY,
        85};

    if (!imencode(
            ".jpg",
            frame,
            jpeg,
            jpegParams)) {

      cerr
          << "JPEG encoding failed\n";

      continue;
    }

    // --------------------------------------------------------
    // Send MJPEG frame
    // --------------------------------------------------------

    string frameHeader =
        "--frame\r\n"
        "Content-Type: image/jpeg\r\n"
        "Content-Length: " +
        to_string(
            jpeg.size()) +
        "\r\n\r\n";

    if (!sendAll(
            clientSocket,
            frameHeader.data(),
            frameHeader.size())) {

      cerr
          << "Viewer disconnected\n";

      close(
          clientSocket);

      clientSocket = -1;

      frameCount++;
      continue;
    }

    if (!sendAll(
            clientSocket,
            jpeg.data(),
            jpeg.size())) {

      cerr
          << "Viewer disconnected\n";

      close(
          clientSocket);

      clientSocket = -1;

      frameCount++;
      continue;
    }

    const char *frameEnd =
        "\r\n";

    if (!sendAll(
            clientSocket,
            frameEnd,
            2)) {

      cerr
          << "Viewer disconnected\n";

      close(
          clientSocket);

      clientSocket = -1;

      frameCount++;
      continue;
    }

    frameCount++;
  }

  // ==========================================================
  // Final benchmark summary
  // ==========================================================

  auto totalRuntimeMs =
      duration_cast<milliseconds>(
          high_resolution_clock::now() -
          startTime)
          .count();

  double effectiveFps =
      totalRuntimeMs > 0
          ? static_cast<double>(
                frameCount) *
                1000.0 /
                static_cast<double>(
                    totalRuntimeMs)
          : 0.0;

  double avgCpuProcessingMs =
      cpuFrameCount > 0
          ? totalCpuProcessingMs /
                cpuFrameCount
          : 0.0;

  double avgGpuKernelMs =
      gpuFrameCount > 0
          ? totalGpuKernelMs /
                gpuFrameCount
          : 0.0;

  cout
      << "\n================ FINAL BENCHMARK SUMMARY ================\n";

  cout
      << "Total Runtime       : "
      << totalRuntimeMs / 1000.0
      << " s\n";

  cout
      << "Total Frames        : "
      << frameCount
      << "\n";

  cout
      << "Overall Throughput  : "
      << effectiveFps
      << " FPS\n";

  cout
      << "CPU Frames          : "
      << cpuFrameCount
      << "\n";

  cout
      << "GPU Frames          : "
      << gpuFrameCount
      << "\n";

  if (cpuFrameCount > 0) {

    cout
        << "Avg CPU Gray+Resize : "
        << avgCpuProcessingMs
        << " ms\n";
  }

  if (gpuFrameCount > 0) {

    cout
        << "Avg GPU Kernel      : "
        << avgGpuKernelMs
        << " ms\n";
  }

  cout
      << "=========================================================\n";

  // ==========================================================
  // Cleanup
  // ==========================================================

  if (clientSocket >= 0) {
    close(
        clientSocket);
  }

  close(
      serverSocket);

  cap.release();

  cleanupOpenCL(
      ocl);

  restoreTerminal();

  return 0;
}