#include <CL/cl.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;
using namespace std::chrono;

// ============================================================
// Configuration
// ============================================================

static const int RECORD_SECONDS = 20;
static const int CAMERA_WIDTH = 1280;
static const int CAMERA_HEIGHT = 720;
static const int CAMERA_FPS = 30;

static const string INPUT_VIDEO = "test_input.mp4";
static const string CPU_OUTPUT_VIDEO = "cpu_output.mp4";
static const string GPU_OUTPUT_VIDEO = "gpu_output.mp4";
static const string BENCHMARK_LOG = "benchmark.log";

// ============================================================
// OpenCL context
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
// CPU/GPU utilization helpers
// ============================================================

struct CpuSnapshot {
    unsigned long long totalUser = 0;
    unsigned long long totalUserLow = 0;
    unsigned long long totalSys = 0;
    unsigned long long totalIdle = 0;
};

CpuSnapshot readCpuSnapshot() {
    CpuSnapshot snap;
    ifstream statFile("/proc/stat");
    string line;

    if (getline(statFile, line) && line.substr(0, 3) == "cpu") {
        stringstream ss(line);
        string cpuLabel;
        ss >> cpuLabel
           >> snap.totalUser
           >> snap.totalUserLow
           >> snap.totalSys
           >> snap.totalIdle;
    }

    return snap;
}

double calculateCpuUsage(const CpuSnapshot &prev, const CpuSnapshot &curr) {
    unsigned long long prevTotal =
        prev.totalUser + prev.totalUserLow + prev.totalSys + prev.totalIdle;

    unsigned long long currTotal =
        curr.totalUser + curr.totalUserLow + curr.totalSys + curr.totalIdle;

    unsigned long long totalDelta = currTotal - prevTotal;
    unsigned long long idleDelta = curr.totalIdle - prev.totalIdle;

    if (totalDelta == 0) {
        return 0.0;
    }

    return
        (1.0 -
         static_cast<double>(idleDelta) /
             static_cast<double>(totalDelta)) *
        100.0;
}

double readGpuUsage() {
    ifstream gpubusyFile("/sys/class/kgsl/kgsl-3d0/gpubusy");

    if (gpubusyFile.is_open()) {
        unsigned long long busyCycles = 0;
        unsigned long long totalCycles = 0;

        gpubusyFile >> busyCycles >> totalCycles;

        if (totalCycles > 0) {
            return
                static_cast<double>(busyCycles) /
                static_cast<double>(totalCycles) *
                100.0;
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

// ============================================================
// Benchmark structures
// ============================================================

enum class PipelineMode {
    CPU,
    GPU
};

struct StageTimings {
    double preprocessMs = 0.0;   // Full CPU or GPU preprocessing operation
    double gpuKernelMs = 0.0;    // GPU kernel only; 0 in CPU mode
    double faceMs = 0.0;         // Haar face detection
    double eyeMs = 0.0;          // Haar eye detection
    double earMs = 0.0;          // Ear position estimation
    double drawMs = 0.0;         // Face/eye/ear annotation only
    double detectionMs = 0.0;    // face + eye + ear
    double totalComputeMs = 0.0;  // preprocess + detection + draw
};

struct BenchmarkStats {
    unsigned long long frames = 0;
    unsigned long long facesDetected = 0;
    unsigned long long eyesDetected = 0;

    double preprocessMsSum = 0.0;
    double gpuKernelMsSum = 0.0;
    double faceMsSum = 0.0;
    double eyeMsSum = 0.0;
    double earMsSum = 0.0;
    double drawMsSum = 0.0;
    double detectionMsSum = 0.0;
    double totalComputeMsSum = 0.0;

    double cpuUsageSum = 0.0;
    double gpuUsageSum = 0.0;
    unsigned long long cpuUsageSamples = 0;
    unsigned long long gpuUsageSamples = 0;

    double wallMs = 0.0;
};

// ============================================================
// OpenCL setup / cleanup
// ============================================================

bool initOpenCL(
    OpenCLContext &ocl,
    const string &kernelFile,
    int width,
    int height) {

    cl_int err = CL_SUCCESS;
    cl_uint numPlatforms = 0;

    err = clGetPlatformIDs(
        1,
        &ocl.platform,
        &numPlatforms);

    if (err != CL_SUCCESS || numPlatforms == 0) {
        cerr << "Could not find OpenCL platform\n";
        return false;
    }

    err = clGetDeviceIDs(
        ocl.platform,
        CL_DEVICE_TYPE_GPU,
        1,
        &ocl.device,
        nullptr);

    if (err != CL_SUCCESS) {
        cerr << "Could not find GPU OpenCL device\n";
        return false;
    }

    ocl.context = clCreateContext(
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

    ocl.queue = clCreateCommandQueue(
        ocl.context,
        ocl.device,
        CL_QUEUE_PROFILING_ENABLE,
        &err);

    if (err != CL_SUCCESS) {
        cerr << "Could not create OpenCL command queue\n";
        return false;
    }

    ifstream file(kernelFile);

    if (!file.is_open()) {
        cerr << "Could not open " << kernelFile << "\n";
        return false;
    }

    string source(
        (istreambuf_iterator<char>(file)),
        istreambuf_iterator<char>());

    const char *sourcePtr = source.c_str();
    size_t sourceLength = source.size();

    ocl.program = clCreateProgramWithSource(
        ocl.context,
        1,
        &sourcePtr,
        &sourceLength,
        &err);

    if (err != CL_SUCCESS) {
        cerr << "Could not create OpenCL program\n";
        return false;
    }

    err = clBuildProgram(
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

        vector<char> buildLog(logSize);

        clGetProgramBuildInfo(
            ocl.program,
            ocl.device,
            CL_PROGRAM_BUILD_LOG,
            logSize,
            buildLog.data(),
            nullptr);

        cerr << "OpenCL Build Error:\n"
             << buildLog.data()
             << "\n";

        return false;
    }

    ocl.kernel = clCreateKernel(
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

    ocl.d_src = clCreateBuffer(
        ocl.context,
        CL_MEM_READ_ONLY,
        srcBytes,
        nullptr,
        &err);

    if (err != CL_SUCCESS) {
        cerr << "Could not create OpenCL source buffer\n";
        return false;
    }

    ocl.d_dst = clCreateBuffer(
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

void cleanupOpenCL(OpenCLContext &ocl) {
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

    ocl = OpenCLContext{};
}

// ============================================================
// Video writer helper
// ============================================================

bool openMp4Writer(
    VideoWriter &writer,
    const string &filename,
    double fps,
    const Size &frameSize) {

    string hardwarePipeline =
        "appsrc ! videoconvert ! v4l2h264enc ! "
        "video/x-h264,profile=baseline ! h264parse ! mp4mux ! "
        "filesink location=" +
        filename;

    writer.open(
        hardwarePipeline,
        CAP_GSTREAMER,
        0,
        fps,
        frameSize,
        true);

    if (writer.isOpened()) {
        return true;
    }

    cerr << "Hardware encoder unavailable for "
         << filename
         << "; trying x264enc fallback...\n";

    string softwarePipeline =
        "appsrc ! videoconvert ! "
        "x264enc tune=zerolatency speed-preset=ultrafast ! "
        "video/x-h264,profile=baseline ! h264parse ! mp4mux ! "
        "filesink location=" +
        filename;

    writer.open(
        softwarePipeline,
        CAP_GSTREAMER,
        0,
        fps,
        frameSize,
        true);

    return writer.isOpened();
}

// ============================================================
// Phase 1: record one reference video
// ============================================================

bool recordReferenceVideo(
    const string &filename,
    int durationSeconds) {

    string inputPipeline =
        "qtiqmmfsrc camera=0 ! "
        "video/x-raw,format=NV12,width=" +
        to_string(CAMERA_WIDTH) +
        ",height=" +
        to_string(CAMERA_HEIGHT) +
        ",framerate=" +
        to_string(CAMERA_FPS) +
        "/1 ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink drop=true sync=false";

    VideoCapture camera(
        inputPipeline,
        CAP_GSTREAMER);

    if (!camera.isOpened()) {
        cerr << "Could not open RB3 camera\n";
        return false;
    }

    Mat frame;

    if (!camera.read(frame) || frame.empty()) {
        cerr << "Could not read first camera frame\n";
        return false;
    }

    VideoWriter writer;

    if (!openMp4Writer(
            writer,
            filename,
            static_cast<double>(CAMERA_FPS),
            frame.size())) {

        cerr << "Could not create "
             << filename
             << "\n";

        return false;
    }

    cout << "\nPHASE 1/3: Recording reference video\n";
    cout << "Recording "
         << durationSeconds
         << " seconds to "
         << filename
         << "...\n";

    auto startTime =
        high_resolution_clock::now();

    unsigned long long recordedFrames = 0;

    while (
        duration_cast<seconds>(
            high_resolution_clock::now() -
            startTime)
                .count() < durationSeconds) {

        if (frame.empty()) {
            break;
        }

        writer.write(frame);
        recordedFrames++;

        if (!camera.read(frame)) {
            break;
        }
    }

    writer.release();
    camera.release();

    if (recordedFrames == 0) {
        cerr << "No frames were recorded\n";
        return false;
    }

    cout << "Reference video saved: "
         << filename
         << "\n";

    return true;
}

// ============================================================
// CPU preprocessing
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
// GPU preprocessing
//
// endToEndMs includes:
//   host -> device transfer
//   kernel execution
//   device -> host transfer
//
// kernelMs is kernel execution only.
// ============================================================

bool processGpu(
    OpenCLContext &ocl,
    const Mat &frame,
    Mat &smallGray,
    int fullWidth,
    int fullHeight,
    int smallWidth,
    int smallHeight,
    double &kernelMs,
    double &endToEndMs) {

    auto gpuStart =
        high_resolution_clock::now();

    Mat contiguousFrame;

    if (frame.isContinuous()) {
        contiguousFrame = frame;
    } else {
        contiguousFrame = frame.clone();
    }

    cl_int err = CL_SUCCESS;

    size_t sourceBytes =
        static_cast<size_t>(fullHeight) *
        contiguousFrame.step;

    err = clEnqueueWriteBuffer(
        ocl.queue,
        ocl.d_src,
        CL_FALSE,
        0,
        sourceBytes,
        contiguousFrame.data,
        0,
        nullptr,
        nullptr);

    if (err != CL_SUCCESS) {
        cerr << "clEnqueueWriteBuffer failed: "
             << err
             << "\n";
        return false;
    }

    int srcStep =
        static_cast<int>(
            contiguousFrame.step);

    int dstStep =
        static_cast<int>(
            smallGray.step);

    err = CL_SUCCESS;

    err |= clSetKernelArg(
        ocl.kernel,
        0,
        sizeof(cl_mem),
        &ocl.d_src);

    err |= clSetKernelArg(
        ocl.kernel,
        1,
        sizeof(cl_mem),
        &ocl.d_dst);

    err |= clSetKernelArg(
        ocl.kernel,
        2,
        sizeof(int),
        &fullWidth);

    err |= clSetKernelArg(
        ocl.kernel,
        3,
        sizeof(int),
        &fullHeight);

    err |= clSetKernelArg(
        ocl.kernel,
        4,
        sizeof(int),
        &srcStep);

    err |= clSetKernelArg(
        ocl.kernel,
        5,
        sizeof(int),
        &dstStep);

    if (err != CL_SUCCESS) {
        cerr << "clSetKernelArg failed: "
             << err
             << "\n";
        return false;
    }

    size_t globalWorkSize[2] = {
        static_cast<size_t>(smallWidth),
        static_cast<size_t>(smallHeight)};

    size_t localWorkSize[2] = {
        16,
        16};

    cl_event kernelEvent = nullptr;

    err = clEnqueueNDRangeKernel(
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
        cerr << "clEnqueueNDRangeKernel failed: "
             << err
             << "\n";
        return false;
    }

    err = clEnqueueReadBuffer(
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
        cerr << "clEnqueueReadBuffer failed: "
             << err
             << "\n";

        clReleaseEvent(
            kernelEvent);

        return false;
    }

    auto gpuEnd =
        high_resolution_clock::now();

    endToEndMs =
        duration_cast<microseconds>(
            gpuEnd - gpuStart)
            .count() /
        1000.0;

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
            kernelEnd -
            kernelStart) *
        1e-6;

    clReleaseEvent(
        kernelEvent);

    return true;
}

// ============================================================
// Draw HUD onto processed output video
// ============================================================

void drawBenchmarkHUD(
    Mat &frame,
    PipelineMode mode,
    const StageTimings &timings,
    double processingFps,
    double cpuUsage,
    double gpuUsage) {

    vector<string> lines;

    stringstream ss;

    ss << "MODE: "
       << (mode == PipelineMode::CPU
               ? "CPU"
               : "GPU-ACCEL");

    lines.push_back(
        ss.str());

    ss.str("");
    ss.clear();

    ss << fixed
       << setprecision(2)
       << "Preprocess: "
       << timings.preprocessMs
       << " ms";

    if (mode == PipelineMode::GPU) {
        ss << " (kernel "
           << timings.gpuKernelMs
           << " ms)";
    }

    lines.push_back(
        ss.str());

    ss.str("");
    ss.clear();

    ss << fixed
       << setprecision(2)
       << "Face: "
       << timings.faceMs
       << " ms | Eye: "
       << timings.eyeMs
       << " ms";

    lines.push_back(
        ss.str());

    ss.str("");
    ss.clear();

    ss << fixed
       << setprecision(2)
       << "Ear: "
       << timings.earMs
       << " ms | Draw: "
       << timings.drawMs
       << " ms";

    lines.push_back(
        ss.str());

    ss.str("");
    ss.clear();

    ss << fixed
       << setprecision(2)
       << "Compute: "
       << timings.totalComputeMs
       << " ms | "
       << setprecision(1)
       << processingFps
       << " FPS";

    lines.push_back(
        ss.str());

    ss.str("");
    ss.clear();

    ss << fixed
       << setprecision(1)
       << "CPU: "
       << cpuUsage
       << "% | GPU: ";

    if (gpuUsage >= 0.0) {
        ss << gpuUsage
           << "%";
    } else {
        ss << "N/A";
    }

    lines.push_back(
        ss.str());

    int fontFace =
        FONT_HERSHEY_SIMPLEX;

    double fontScale =
        0.45;

    int thickness =
        1;

    int lineSpacing =
        20;

    int margin =
        10;

    int maxTextWidth =
        0;

    for (const auto &line : lines) {
        int baseline = 0;

        Size textSize =
            getTextSize(
                line,
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
        frame.cols -
        boxWidth -
        10;

    int boxY =
        10;

    Rect hudRect(
        boxX,
        boxY,
        boxWidth,
        boxHeight);

    if (
        hudRect.x >= 0 &&
        hudRect.y >= 0 &&
        hudRect.x + hudRect.width <= frame.cols &&
        hudRect.y + hudRect.height <= frame.rows) {

        Mat roi =
            frame(
                hudRect);

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
            0.75,
            roi,
            0.25,
            0,
            roi);

        rectangle(
            frame,
            hudRect,
            Scalar(
                90,
                90,
                90),
            1);
    }

    int textY =
        boxY +
        margin +
        12;

    for (const auto &line : lines) {
        putText(
            frame,
            line,
            Point(
                boxX +
                    margin,
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
// Run one benchmark pass over the exact same recorded video
// ============================================================

bool runBenchmark(
    const string &inputFilename,
    const string &outputFilename,
    PipelineMode mode,
    CascadeClassifier &faceCascade,
    CascadeClassifier &eyeCascade,
    OpenCLContext *ocl,
    ofstream &benchmarkLog,
    BenchmarkStats &stats) {

    VideoCapture input(
        inputFilename);

    if (!input.isOpened()) {
        cerr << "Could not open "
             << inputFilename
             << " for benchmark\n";
        return false;
    }

    double sourceFps =
        input.get(
            CAP_PROP_FPS);

    if (sourceFps <= 0.0) {
        sourceFps =
            static_cast<double>(
                CAMERA_FPS);
    }

    int fullWidth =
        static_cast<int>(
            input.get(
                CAP_PROP_FRAME_WIDTH));

    int fullHeight =
        static_cast<int>(
            input.get(
                CAP_PROP_FRAME_HEIGHT));

    if (
        fullWidth <= 0 ||
        fullHeight <= 0) {

        Mat probeFrame;

        if (!input.read(probeFrame) || probeFrame.empty()) {
            cerr << "Invalid input video dimensions\n";
            return false;
        }

        fullWidth =
            probeFrame.cols;

        fullHeight =
            probeFrame.rows;

        input.set(
            CAP_PROP_POS_FRAMES,
            0);
    }

    int smallWidth =
        fullWidth /
        2;

    int smallHeight =
        fullHeight /
        2;

    Mat gray;

    Mat smallGray(
        smallHeight,
        smallWidth,
        CV_8UC1);

    const double invScale =
        2.0;

    VideoWriter output;

    if (!openMp4Writer(
            output,
            outputFilename,
            sourceFps,
            Size(
                fullWidth,
                fullHeight))) {

        cerr << "Could not create "
             << outputFilename
             << "\n";

        return false;
    }

    const string modeName =
        mode == PipelineMode::CPU
            ? "CPU"
            : "GPU-ACCEL";

    cout << "\n"
         << (mode == PipelineMode::CPU
                 ? "PHASE 2/3: CPU benchmark"
                 : "PHASE 3/3: GPU-accelerated benchmark")
         << "\n";

    cout << "Input : "
         << inputFilename
         << "\n";

    cout << "Output: "
         << outputFilename
         << "\n";

    benchmarkLog
        << "\n============================================================\n"
        << modeName
        << " PIPELINE\n"
        << "============================================================\n";

    if (mode == PipelineMode::CPU) {
        benchmarkLog
            << "Preprocessing: CPU cvtColor + CPU resize\n";
    } else {
        benchmarkLog
            << "Preprocessing: OpenCL GPU grayscale + 2x downscale\n"
            << "IMPORTANT: Haar face/eye detection remains CPU in this code.\n";
    }

    benchmarkLog
        << "Frame,Preprocess_ms,GPU_Kernel_ms,Face_ms,Eye_ms,Ear_ms,Draw_ms,"
        << "Detection_ms,Total_Compute_ms,Processing_FPS,Faces,Eyes\n";

    CpuSnapshot lastCpuSnapshot =
        readCpuSnapshot();

    auto lastUsageTime =
        high_resolution_clock::now();

    double currentCpuUsage =
        0.0;

    double currentGpuUsage =
        0.0;

    auto wallStart =
        high_resolution_clock::now();

    Mat frame;

    while (
        input.read(
            frame)) {

        if (frame.empty()) {
            break;
        }

        StageTimings timings;

        // --------------------------------------------------------
        // Preprocessing
        // --------------------------------------------------------

        if (mode == PipelineMode::CPU) {

            timings.preprocessMs =
                processCpu(
                    frame,
                    gray,
                    smallGray,
                    smallWidth,
                    smallHeight);

        } else {

            if (ocl == nullptr) {
                cerr << "GPU benchmark requested without OpenCL context\n";
                return false;
            }

            double endToEndMs =
                0.0;

            if (!processGpu(
                    *ocl,
                    frame,
                    smallGray,
                    fullWidth,
                    fullHeight,
                    smallWidth,
                    smallHeight,
                    timings.gpuKernelMs,
                    endToEndMs)) {

                cerr << "GPU preprocessing failed\n";
                return false;
            }

            timings.preprocessMs =
                endToEndMs;
        }

        // --------------------------------------------------------
        // Face detection
        // --------------------------------------------------------

        vector<Rect> faces;

        auto faceStart =
            high_resolution_clock::now();

        faceCascade.detectMultiScale(
            smallGray,
            faces,
            1.1,
            4,
            0,
            Size(
                30,
                30));

        timings.faceMs =
            duration_cast<microseconds>(
                high_resolution_clock::now() -
                faceStart)
                .count() /
            1000.0;

        int processedFaceCount =
            0;

        int eyeCount =
            0;

        // Keep the same behavior as the uploaded code:
        // only process the first detected face.
        if (!faces.empty()) {

            const Rect &smallFace =
                faces.front();

            processedFaceCount =
                1;

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

            face &=
                Rect(
                    0,
                    0,
                    frame.cols,
                    frame.rows);

            if (
                face.width > 0 &&
                face.height > 0) {

                // ------------------------------------------------
                // Eye detection
                // ------------------------------------------------

                Mat faceROI =
                    smallGray(
                        smallFace);

                vector<Rect> eyes;

                auto eyeStart =
                    high_resolution_clock::now();

                eyeCascade.detectMultiScale(
                    faceROI,
                    eyes,
                    1.1,
                    4,
                    0,
                    Size(
                        15,
                        15));

                timings.eyeMs =
                    duration_cast<microseconds>(
                        high_resolution_clock::now() -
                        eyeStart)
                        .count() /
                    1000.0;

                eyeCount =
                    static_cast<int>(
                        eyes.size());

                // ------------------------------------------------
                // Ear position estimation
                // ------------------------------------------------

                int earWidth = 0;
                int earHeight = 0;
                int earY = 0;
                Rect leftEarRect;
                Rect rightEarRect;

                auto earStart =
                    high_resolution_clock::now();

                earWidth =
                    static_cast<int>(
                        face.width *
                        0.18);

                earHeight =
                    static_cast<int>(
                        face.height *
                        0.35);

                earY =
                    face.y +
                    static_cast<int>(
                        face.height *
                        0.28);

                if (
                    earWidth > 0 &&
                    earHeight > 0) {

                    leftEarRect =
                        Rect(
                            max(
                                0,
                                face.x -
                                    static_cast<int>(
                                        earWidth *
                                        0.6)),
                            earY,
                            earWidth,
                            earHeight);

                    rightEarRect =
                        Rect(
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
                }

                timings.earMs =
                    duration_cast<microseconds>(
                        high_resolution_clock::now() -
                        earStart)
                        .count() /
                    1000.0;

                // ------------------------------------------------
                // Draw annotations
                // ------------------------------------------------

                auto drawStart =
                    high_resolution_clock::now();

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
                            face.y -
                                5)),
                    FONT_HERSHEY_SIMPLEX,
                    0.5,
                    Scalar(
                        0,
                        0,
                        255),
                    1);

                for (
                    const Rect &smallEye :
                    eyes) {

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

                    if (
                        eyeGlobal.width <= 0 ||
                        eyeGlobal.height <= 0) {
                        continue;
                    }

                    // Anatomical perspective:
                    // image-left = subject's RIGHT eye
                    // image-right = subject's LEFT eye
                    double eyeCenterX =
                        smallEye.x +
                        smallEye.width *
                            0.5;

                    double faceCenterX =
                        smallFace.width *
                        0.5;

                    string eyeLabel =
                        eyeCenterX <
                                faceCenterX
                            ? "R Eye"
                            : "L Eye";

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
                        eyeLabel,
                        Point(
                            eyeGlobal.x,
                            max(
                                0,
                                eyeGlobal.y -
                                    4)),
                        FONT_HERSHEY_SIMPLEX,
                        0.4,
                        Scalar(
                            0,
                            255,
                            0),
                        1);
                }

                if (
                    leftEarRect.width > 0 &&
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
                                leftEarRect.y -
                                    4)),
                        FONT_HERSHEY_SIMPLEX,
                        0.4,
                        Scalar(
                            255,
                            0,
                            0),
                        1);
                }

                if (
                    rightEarRect.width > 0 &&
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
                                rightEarRect.y -
                                    4)),
                        FONT_HERSHEY_SIMPLEX,
                        0.4,
                        Scalar(
                            255,
                            255,
                            0),
                        1);
                }

                timings.drawMs =
                    duration_cast<microseconds>(
                        high_resolution_clock::now() -
                        drawStart)
                        .count() /
                    1000.0;
            }
        }

        // --------------------------------------------------------
        // Final per-frame compute metrics
        // --------------------------------------------------------

        timings.detectionMs =
            timings.faceMs +
            timings.eyeMs +
            timings.earMs;

        timings.totalComputeMs =
            timings.preprocessMs +
            timings.detectionMs +
            timings.drawMs;

        double processingFps =
            timings.totalComputeMs >
                    0.0
                ? 1000.0 /
                      timings.totalComputeMs
                : 0.0;

        // --------------------------------------------------------
        // Update accumulated statistics
        // --------------------------------------------------------

        stats.frames++;

        stats.facesDetected +=
            static_cast<unsigned long long>(
                processedFaceCount);

        stats.eyesDetected +=
            static_cast<unsigned long long>(
                eyeCount);

        stats.preprocessMsSum +=
            timings.preprocessMs;

        stats.gpuKernelMsSum +=
            timings.gpuKernelMs;

        stats.faceMsSum +=
            timings.faceMs;

        stats.eyeMsSum +=
            timings.eyeMs;

        stats.earMsSum +=
            timings.earMs;

        stats.drawMsSum +=
            timings.drawMs;

        stats.detectionMsSum +=
            timings.detectionMs;

        stats.totalComputeMsSum +=
            timings.totalComputeMs;

        // --------------------------------------------------------
        // Sample system utilization every 500 ms
        // --------------------------------------------------------

        auto now =
            high_resolution_clock::now();

        double usageIntervalSec =
            duration_cast<milliseconds>(
                now -
                lastUsageTime)
                .count() /
            1000.0;

        if (usageIntervalSec >= 0.5) {

            CpuSnapshot currentCpuSnapshot =
                readCpuSnapshot();

            currentCpuUsage =
                calculateCpuUsage(
                    lastCpuSnapshot,
                    currentCpuSnapshot);

            currentGpuUsage =
                readGpuUsage();

            stats.cpuUsageSum +=
                currentCpuUsage;

            stats.cpuUsageSamples++;

            if (currentGpuUsage >= 0.0) {
                stats.gpuUsageSum +=
                    currentGpuUsage;

                stats.gpuUsageSamples++;
            }

            lastCpuSnapshot =
                currentCpuSnapshot;

            lastUsageTime =
                now;
        }

        // --------------------------------------------------------
        // Per-frame benchmark log
        // --------------------------------------------------------

        benchmarkLog
            << stats.frames
            << ","
            << fixed
            << setprecision(3)
            << timings.preprocessMs
            << ","
            << timings.gpuKernelMs
            << ","
            << timings.faceMs
            << ","
            << timings.eyeMs
            << ","
            << timings.earMs
            << ","
            << timings.drawMs
            << ","
            << timings.detectionMs
            << ","
            << timings.totalComputeMs
            << ","
            << processingFps
            << ","
            << processedFaceCount
            << ","
            << eyeCount
            << "\n";

        // HUD and output encoding are intentionally AFTER the
        // benchmark timing so they do not inflate compute time.
        drawBenchmarkHUD(
            frame,
            mode,
            timings,
            processingFps,
            currentCpuUsage,
            currentGpuUsage);

        output.write(
            frame);
    }

    stats.wallMs =
        duration_cast<milliseconds>(
            high_resolution_clock::now() -
            wallStart)
            .count();

    output.release();
    input.release();

    if (stats.frames == 0) {
        cerr << "No frames processed in "
             << modeName
             << " benchmark\n";

        return false;
    }

    double frameCount =
        static_cast<double>(
            stats.frames);

    double avgComputeMs =
        stats.totalComputeMsSum /
        frameCount;

    double computeFps =
        avgComputeMs >
                0.0
            ? 1000.0 /
                  avgComputeMs
            : 0.0;

    double wallThroughputFps =
        stats.wallMs >
                0.0
            ? frameCount *
                  1000.0 /
                  stats.wallMs
            : 0.0;

    benchmarkLog
        << "\n"
        << modeName
        << " SUMMARY\n"
        << fixed
        << setprecision(3)
        << "Frames processed              : "
        << stats.frames
        << "\n"
        << "Average preprocess            : "
        << stats.preprocessMsSum /
               frameCount
        << " ms/frame\n";

    if (mode == PipelineMode::GPU) {
        benchmarkLog
            << "Average GPU kernel            : "
            << stats.gpuKernelMsSum /
                   frameCount
            << " ms/frame\n";
    }

    benchmarkLog
        << "Average face detection        : "
        << stats.faceMsSum /
               frameCount
        << " ms/frame\n"
        << "Average eye detection         : "
        << stats.eyeMsSum /
               frameCount
        << " ms/frame\n"
        << "Average ear estimation        : "
        << stats.earMsSum /
               frameCount
        << " ms/frame\n"
        << "Average drawing               : "
        << stats.drawMsSum /
               frameCount
        << " ms/frame\n"
        << "Average detection total       : "
        << stats.detectionMsSum /
               frameCount
        << " ms/frame\n"
        << "Average total compute         : "
        << avgComputeMs
        << " ms/frame\n"
        << "Compute throughput            : "
        << computeFps
        << " FPS\n"
        << "Wall throughput               : "
        << wallThroughputFps
        << " FPS"
        << " (includes decode + output encode)\n"
        << "Average detections/frame      : "
        << static_cast<double>(
               stats.facesDetected) /
               frameCount
        << " face | "
        << static_cast<double>(
               stats.eyesDetected) /
               frameCount
        << " eyes\n";

    if (stats.cpuUsageSamples > 0) {
        benchmarkLog
            << "Average sampled CPU usage     : "
            << stats.cpuUsageSum /
                   static_cast<double>(
                       stats.cpuUsageSamples)
            << "%\n";
    }

    if (stats.gpuUsageSamples > 0) {
        benchmarkLog
            << "Average sampled GPU usage     : "
            << stats.gpuUsageSum /
                   static_cast<double>(
                       stats.gpuUsageSamples)
            << "%\n";
    }

    benchmarkLog.flush();

    cout << modeName
         << " pass complete: "
         << stats.frames
         << " frames, "
         << fixed
         << setprecision(1)
         << computeFps
         << " compute FPS\n";

    return true;
}

// ============================================================
// Final CPU vs GPU comparison
// ============================================================

void writeComparison(
    ofstream &benchmarkLog,
    const BenchmarkStats &cpu,
    const BenchmarkStats &gpu) {

    if (
        cpu.frames == 0 ||
        gpu.frames == 0) {
        return;
    }

    double cpuFrames =
        static_cast<double>(
            cpu.frames);

    double gpuFrames =
        static_cast<double>(
            gpu.frames);

    double cpuPreprocess =
        cpu.preprocessMsSum /
        cpuFrames;

    double gpuPreprocess =
        gpu.preprocessMsSum /
        gpuFrames;

    double cpuDetection =
        cpu.detectionMsSum /
        cpuFrames;

    double gpuDetection =
        gpu.detectionMsSum /
        gpuFrames;

    double cpuTotal =
        cpu.totalComputeMsSum /
        cpuFrames;

    double gpuTotal =
        gpu.totalComputeMsSum /
        gpuFrames;

    double cpuComputeFps =
        cpuTotal > 0.0
            ? 1000.0 /
                  cpuTotal
            : 0.0;

    double gpuComputeFps =
        gpuTotal > 0.0
            ? 1000.0 /
                  gpuTotal
            : 0.0;

    benchmarkLog
        << "\n============================================================\n"
        << "CPU VS GPU-ACCELERATED PIPELINE COMPARISON\n"
        << "============================================================\n"
        << fixed
        << setprecision(3)
        << "Same recorded input video     : "
        << INPUT_VIDEO
        << "\n"
        << "CPU frames processed          : "
        << cpu.frames
        << "\n"
        << "GPU frames processed          : "
        << gpu.frames
        << "\n\n"
        << "CPU avg preprocess            : "
        << cpuPreprocess
        << " ms/frame\n"
        << "GPU avg preprocess end-to-end : "
        << gpuPreprocess
        << " ms/frame\n"
        << "GPU avg kernel only           : "
        << gpu.gpuKernelMsSum /
               gpuFrames
        << " ms/frame\n";

    if (gpuPreprocess > 0.0) {
        benchmarkLog
            << "Preprocessing speedup         : "
            << cpuPreprocess /
                   gpuPreprocess
            << "x\n";
    }

    benchmarkLog
        << "\nCPU avg detection             : "
        << cpuDetection
        << " ms/frame\n"
        << "GPU-pass avg detection        : "
        << gpuDetection
        << " ms/frame\n"
        << "(Haar detection is CPU in BOTH passes.)\n\n"
        << "CPU avg total compute         : "
        << cpuTotal
        << " ms/frame\n"
        << "GPU avg total compute         : "
        << gpuTotal
        << " ms/frame\n"
        << "CPU compute throughput        : "
        << cpuComputeFps
        << " FPS\n"
        << "GPU compute throughput        : "
        << gpuComputeFps
        << " FPS\n";

    if (gpuTotal > 0.0) {
        benchmarkLog
            << "Full-pipeline speedup         : "
            << cpuTotal /
                   gpuTotal
            << "x\n";
    }

    benchmarkLog
        << "============================================================\n";

    benchmarkLog.flush();
}

// ============================================================
// Main
// ============================================================

int main() {

    // ----------------------------------------------------------
    // Load Haar cascades once
    // ----------------------------------------------------------

    CascadeClassifier faceCascade;
    CascadeClassifier eyeCascade;

    string cascadePath =
        "/usr/share/opencv4/haarcascades/";

    if (
        !faceCascade.load(
            cascadePath +
            "haarcascade_frontalface_default.xml") ||
        !eyeCascade.load(
            cascadePath +
            "haarcascade_eye.xml")) {

        cerr << "Could not load Haar cascade classifiers\n";
        return 1;
    }

    // ----------------------------------------------------------
    // Create benchmark log
    // ----------------------------------------------------------

    ofstream benchmarkLog(
        BENCHMARK_LOG,
        ios::out |
            ios::trunc);

    if (!benchmarkLog.is_open()) {
        cerr << "Could not open "
             << BENCHMARK_LOG
             << "\n";
        return 1;
    }

    benchmarkLog
        << "CPU VS GPU VIDEO PROCESSING TEST BENCH\n"
        << "Reference video is recorded once, then replayed through both pipelines.\n"
        << "CPU pipeline: CPU grayscale/downscale + CPU Haar detection.\n"
        << "GPU pipeline: OpenCL grayscale/downscale + CPU Haar detection.\n"
        << "The same recorded frames are used for both tests.\n";

    // ----------------------------------------------------------
    // Phase 1: record input once
    // ----------------------------------------------------------

    if (!recordReferenceVideo(
            INPUT_VIDEO,
            RECORD_SECONDS)) {

        return 1;
    }

    // ----------------------------------------------------------
    // Read dimensions from the recorded source
    // ----------------------------------------------------------

    VideoCapture sourceInfo(
        INPUT_VIDEO);

    if (!sourceInfo.isOpened()) {
        cerr << "Could not reopen "
             << INPUT_VIDEO
             << "\n";
        return 1;
    }

    int fullWidth =
        static_cast<int>(
            sourceInfo.get(
                CAP_PROP_FRAME_WIDTH));

    int fullHeight =
        static_cast<int>(
            sourceInfo.get(
                CAP_PROP_FRAME_HEIGHT));

    if (
        fullWidth <= 0 ||
        fullHeight <= 0) {

        Mat probeFrame;

        if (!sourceInfo.read(probeFrame) || probeFrame.empty()) {
            cerr << "Could not determine recorded video dimensions\n";
            return 1;
        }

        fullWidth =
            probeFrame.cols;

        fullHeight =
            probeFrame.rows;
    }

    sourceInfo.release();

    // ----------------------------------------------------------
    // Phase 2: CPU pass
    // ----------------------------------------------------------

    BenchmarkStats cpuStats;

    if (!runBenchmark(
            INPUT_VIDEO,
            CPU_OUTPUT_VIDEO,
            PipelineMode::CPU,
            faceCascade,
            eyeCascade,
            nullptr,
            benchmarkLog,
            cpuStats)) {

        return 1;
    }

    // ----------------------------------------------------------
    // Initialize GPU only after CPU pass
    // ----------------------------------------------------------

    OpenCLContext ocl;

    if (!initOpenCL(
            ocl,
            "kernel.cl",
            fullWidth,
            fullHeight)) {

        cerr << "Failed to initialize OpenCL GPU pipeline\n";
        return 1;
    }

    // ----------------------------------------------------------
    // Phase 3: GPU-accelerated pass
    // ----------------------------------------------------------

    BenchmarkStats gpuStats;

    bool gpuSuccess =
        runBenchmark(
            INPUT_VIDEO,
            GPU_OUTPUT_VIDEO,
            PipelineMode::GPU,
            faceCascade,
            eyeCascade,
            &ocl,
            benchmarkLog,
            gpuStats);

    cleanupOpenCL(
        ocl);

    if (!gpuSuccess) {
        return 1;
    }

    // ----------------------------------------------------------
    // Final comparison
    // ----------------------------------------------------------

    writeComparison(
        benchmarkLog,
        cpuStats,
        gpuStats);

    benchmarkLog.close();

    cout << "\nTEST BENCH COMPLETE\n";
    cout << "Reference video : "
         << INPUT_VIDEO
         << "\n";
    cout << "CPU output      : "
         << CPU_OUTPUT_VIDEO
         << "\n";
    cout << "GPU output      : "
         << GPU_OUTPUT_VIDEO
         << "\n";
    cout << "Benchmark log   : "
         << BENCHMARK_LOG
         << "\n";

    return 0;
}