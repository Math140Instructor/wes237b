#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <CL/cl.h>

using namespace cv;
using namespace std;
using namespace std::chrono;

// OpenCL Context
struct OpenCLContext {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem d_src;
    cl_mem d_dst;
};

// System Utilization Metrics
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
    if (getline(statFile, line)) {
        if (line.substr(0, 3) == "cpu") {
            stringstream ss(line);
            string cpuLabel;
            ss >> cpuLabel >> snap.totalUser >> snap.totalUserLow >> snap.totalSys >> snap.totalIdle;
        }
    }
    return snap;
}

double calculateCpuUsage(const CpuSnapshot &prev, const CpuSnapshot &curr) {
    unsigned long long prevTotal = prev.totalUser + prev.totalUserLow + prev.totalSys + prev.totalIdle;
    unsigned long long currTotal = curr.totalUser + curr.totalUserLow + curr.totalSys + curr.totalIdle;
    unsigned long long totalDelta = currTotal - prevTotal;

    unsigned long long prevIdle = prev.totalIdle;
    unsigned long long currIdle = curr.totalIdle;
    unsigned long long idleDelta = currIdle - prevIdle;

    if (totalDelta == 0) return 0.0;
    return (1.0 - static_cast<double>(idleDelta) / totalDelta) * 100.0;
}

double readGpuUsage() {
    ifstream gpubusyFile("/sys/class/kgsl/kgsl-3d0/gpubusy");
    if (gpubusyFile.is_open()) {
        unsigned long long busyCycles = 0, totalCycles = 0;
        gpubusyFile >> busyCycles >> totalCycles;
        if (totalCycles > 0) {
            return (static_cast<double>(busyCycles) / totalCycles) * 100.0;
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

// Stage Profiling Metrics
struct StageTimings {
    double gpuPreprocMs = 0.0;
    double cpuFaceMs = 0.0;
    double cpuEyeMs = 0.0;
    double cpuEarMs = 0.0;
    double cpuDrawMs = 0.0;
    double totalFrameCpuMs = 0.0;
};

bool initOpenCL(OpenCLContext &ocl, const string &kernelFile, int width, int height) {
    cl_uint numPlatforms;
    cl_int err = clGetPlatformIDs(1, &ocl.platform, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) return false;

    err = clGetDeviceIDs(ocl.platform, CL_DEVICE_TYPE_GPU, 1, &ocl.device, NULL);
    if (err != CL_SUCCESS) {
        err = clGetDeviceIDs(ocl.platform, CL_DEVICE_TYPE_DEFAULT, 1, &ocl.device, NULL);
        if (err != CL_SUCCESS) return false;
    }

    ocl.context = clCreateContext(NULL, 1, &ocl.device, NULL, NULL, &err);
    if (err != CL_SUCCESS) return false;

    ocl.queue = clCreateCommandQueue(ocl.context, ocl.device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) return false;

    ifstream file(kernelFile);
    if (!file.is_open()) {
        cerr << "Could not open " << kernelFile << "\n";
        return false;
    }
    string srcStr((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    const char *src = srcStr.c_str();
    size_t length = srcStr.length();

    ocl.program = clCreateProgramWithSource(ocl.context, 1, &src, &length, &err);
    if (err != CL_SUCCESS) return false;

    err = clBuildProgram(ocl.program, 1, &ocl.device, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(ocl.program, ocl.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        vector<char> log(logSize);
        clGetProgramBuildInfo(ocl.program, ocl.device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), NULL);
        cerr << "OpenCL Build Error:\n" << log.data() << "\n";
        return false;
    }

    ocl.kernel = clCreateKernel(ocl.program, "bgr_to_gray_downscale_2x", &err);
    if (err != CL_SUCCESS) return false;

    size_t srcBytes = width * height * 3 * sizeof(unsigned char);
    size_t dstBytes = (width / 2) * (height / 2) * sizeof(unsigned char);

    ocl.d_src = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY, srcBytes, NULL, &err);
    ocl.d_dst = clCreateBuffer(ocl.context, CL_MEM_WRITE_ONLY, dstBytes, NULL, &err);
    if (err != CL_SUCCESS) return false;

    return true;
}

void cleanupOpenCL(OpenCLContext &ocl) {
    if (ocl.d_src) clReleaseMemObject(ocl.d_src);
    if (ocl.d_dst) clReleaseMemObject(ocl.d_dst);
    if (ocl.kernel) clReleaseKernel(ocl.kernel);
    if (ocl.program) clReleaseProgram(ocl.program);
    if (ocl.queue) clReleaseCommandQueue(ocl.queue);
    if (ocl.context) clReleaseContext(ocl.context);
}

// Function to render granular stage telemetry in top-right HUD
void drawDetailedHUD(Mat &img, double cpuUsage, double gpuUsage, const StageTimings &t, double instantFps, int frameIdx, double elapsedSec) {
    vector<string> lines;
   
    stringstream ssSys;
    ssSys << fixed << setprecision(1) << "CPU: " << cpuUsage << "% | GPU: " << (gpuUsage >= 0.0 ? to_string((int)gpuUsage) + "%" : "N/A");
    lines.push_back(ssSys.str());

    stringstream ssFps;
    ssFps << fixed << setprecision(1) << "FPS: " << instantFps << " | T: " << elapsedSec << "s (F:" << frameIdx << ")";
    lines.push_back(ssFps.str());

    stringstream ssGpu;
    ssGpu << fixed << setprecision(2) << "GPU Preproc: " << t.gpuPreprocMs << " ms";
    lines.push_back(ssGpu.str());

    stringstream ssFace;
    ssFace << fixed << setprecision(2) << "CPU Face   : " << t.cpuFaceMs << " ms";
    lines.push_back(ssFace.str());

    stringstream ssEye;
    ssEye << fixed << setprecision(2) << "CPU Eye    : " << t.cpuEyeMs << " ms";
    lines.push_back(ssEye.str());

    stringstream ssEar;
    ssEar << fixed << setprecision(2) << "CPU Ear    : " << t.cpuEarMs << " ms";
    lines.push_back(ssEar.str());

    stringstream ssTot;
    ssTot << fixed << setprecision(2) << "CPU Tot/Frm: " << t.totalFrameCpuMs << " ms";
    lines.push_back(ssTot.str());

    int fontFace = FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.45;
    int thickness = 1;
    int lineSpacing = 19;
    int margin = 10;

    int maxTextWidth = 0;
    for (const auto &text : lines) {
        int baseline = 0;
        Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
        if (textSize.width > maxTextWidth) {
            maxTextWidth = textSize.width;
        }
    }

    int boxWidth = maxTextWidth + (margin * 2);
    int boxHeight = (lines.size() * lineSpacing) + margin;
    int boxX = img.cols - boxWidth - 10;
    int boxY = 10;

    Rect hudRect(boxX, boxY, boxWidth, boxHeight);
    if (hudRect.x >= 0 && hudRect.y >= 0 && hudRect.x + hudRect.width <= img.cols && hudRect.y + hudRect.height <= img.rows) {
        Mat roi = img(hudRect);
        Mat overlay;
        roi.copyTo(overlay);
        rectangle(overlay, Rect(0, 0, boxWidth, boxHeight), Scalar(15, 15, 15), FILLED);
        addWeighted(overlay, 0.75, roi, 0.25, 0, roi);
        rectangle(img, hudRect, Scalar(90, 90, 90), 1);
    }

    int textY = boxY + margin + 12;
    for (size_t i = 0; i < lines.size(); ++i) {
        Scalar color = (i < 2) ? Scalar(0, 255, 255) : (i == 2 ? Scalar(255, 200, 0) : Scalar(0, 255, 0));
        putText(img, lines[i], Point(boxX + margin, textY), fontFace, fontScale, color, thickness, LINE_AA);
        textY += lineSpacing;
    }
}

int main() {
    string inputPipeline = "qtiqmmfsrc camera=0 ! "
                           "video/x-raw,format=NV12,width=1280,height=720,framerate=30/1 ! "
                           "videoconvert ! "
                           "video/x-raw,format=BGR ! "
                           "appsink drop=true sync=false";

    VideoCapture cap(inputPipeline, CAP_GSTREAMER);
    if (!cap.isOpened()) {
        cerr << "Could not open RB3 camera\n";
        return 1;
    }
    cout << "RB3 camera opened\n";

    CascadeClassifier faceCascade, eyeCascade;
    string cascadePath = "/usr/share/opencv4/haarcascades/";

    if (!faceCascade.load(cascadePath + "haarcascade_frontalface_default.xml") ||
        !eyeCascade.load(cascadePath + "haarcascade_eye.xml")) {
        cerr << "Could not load cascade classifiers\n";
        return 1;
    }

    Mat frame;
    if (!cap.read(frame) || frame.empty()) {
        cerr << "Could not read first camera frame\n";
        return 1;
    }

    int fullWidth = frame.cols;
    int fullHeight = frame.rows;
    int smallWidth = fullWidth / 2;
    int smallHeight = fullHeight / 2;

    OpenCLContext ocl;
    if (!initOpenCL(ocl, "kernel.cl", fullWidth, fullHeight)) {
        cerr << "Failed to initialize OpenCL GPU pipeline from kernel.cl\n";
        return 1;
    }
    cout << "OpenCL kernel pipeline initialized on GPU.\n";

    ofstream benchmarkLog("benchmark.log", ios::out | ios::trunc);
    if (!benchmarkLog.is_open()) {
        cerr << "Could not open benchmark.log\n";
        cleanupOpenCL(ocl);
        return 1;
    }

    const int targetDurationSeconds = 20;
    int frameCount = 0;
    vector<Mat> frameBuffer;
    frameBuffer.reserve(targetDurationSeconds * 30);

    Mat smallGray(smallHeight, smallWidth, CV_8UC1);
    const double invScale = 2.0;

    // Benchmark Trackers
    CpuSnapshot lastCpuSnap = readCpuSnapshot();
    auto lastBenchmarkTime = high_resolution_clock::now();
    double currentCpuPct = 0.0;
    double currentGpuPct = 0.0;

    // Cumulative Stage Timers
    double totalGpuPreprocMs = 0.0;
    double totalCpuFaceMs    = 0.0;
    double totalCpuEyeMs     = 0.0;
    double totalCpuEarMs     = 0.0;
    double totalCpuDrawMs    = 0.0;
    double totalCpuActiveMs  = 0.0;

    cout << "\n=================================== BENCHMARK MONITOR ===================================\n";
    cout << "Recording for " << targetDurationSeconds << " seconds in 1:1 real-time...\n";
    cout << "Benchmark metrics will be written to benchmark.log\n";
    cout << "-----------------------------------------------------------------------------------------\n";

    benchmarkLog << "=================================== BENCHMARK MONITOR ===================================\n";
    benchmarkLog << "Pipeline: GPU grayscale/downscale + CPU face/eye detection + CPU ear estimation/drawing\n";
    benchmarkLog << "Recording duration: " << targetDurationSeconds << " seconds\n";
    benchmarkLog << "-----------------------------------------------------------------------------------------\n";

    auto startTime = high_resolution_clock::now();
    auto lastFrameTimestamp = startTime;

    while (duration_cast<seconds>(high_resolution_clock::now() - startTime).count() < targetDurationSeconds) {
        auto frameLoopStart = high_resolution_clock::now();
        StageTimings timings;

        if (!cap.read(frame) || frame.empty()) {
            cerr << "Frame capture error.\n";
            break;
        }

        // --- STAGE 1: GPU PREPROCESSING (OpenCL Kernel) ---
        clEnqueueWriteBuffer(ocl.queue, ocl.d_src, CL_FALSE, 0, fullWidth * fullHeight * 3, frame.data, 0, NULL, NULL);

        int srcStep = (int)frame.step;
        int dstStep = (int)smallGray.step;
        clSetKernelArg(ocl.kernel, 0, sizeof(cl_mem), &ocl.d_src);
        clSetKernelArg(ocl.kernel, 1, sizeof(cl_mem), &ocl.d_dst);
        clSetKernelArg(ocl.kernel, 2, sizeof(int), &fullWidth);
        clSetKernelArg(ocl.kernel, 3, sizeof(int), &fullHeight);
        clSetKernelArg(ocl.kernel, 4, sizeof(int), &srcStep);
        clSetKernelArg(ocl.kernel, 5, sizeof(int), &dstStep);

        cl_event kernelEvent;
        size_t globalWorkSize[2] = { (size_t)smallWidth, (size_t)smallHeight };
        size_t localWorkSize[2]  = { 16, 16 };
        clEnqueueNDRangeKernel(ocl.queue, ocl.kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &kernelEvent);
        clEnqueueReadBuffer(ocl.queue, ocl.d_dst, CL_TRUE, 0, smallWidth * smallHeight, smallGray.data, 0, NULL, NULL);

        clWaitForEvents(1, &kernelEvent);
        cl_ulong kStart = 0, kEnd = 0;
        clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kStart, NULL);
        clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kEnd, NULL);
        timings.gpuPreprocMs = (kEnd - kStart) * 1e-6;
        totalGpuPreprocMs += timings.gpuPreprocMs;
        clReleaseEvent(kernelEvent);

        // --- STAGE 2: CPU FACE DETECTION ---
        auto tFaceStart = high_resolution_clock::now();
        vector<Rect> faces;
        faceCascade.detectMultiScale(smallGray, faces, 1.1, 4, 0, Size(30, 30));
        timings.cpuFaceMs = duration_cast<microseconds>(high_resolution_clock::now() - tFaceStart).count() / 1000.0;
        totalCpuFaceMs += timings.cpuFaceMs;

        // Stage 3 & 4: Eyes & Ears
        timings.cpuEyeMs = 0.0;
        timings.cpuEarMs = 0.0;
        timings.cpuDrawMs = 0.0;

        for (const Rect &sFace : faces) {
            auto tDrawStart = high_resolution_clock::now();
            Rect face(cvRound(sFace.x * invScale),
                      cvRound(sFace.y * invScale),
                      cvRound(sFace.width * invScale),
                      cvRound(sFace.height * invScale));

            rectangle(frame, face, Scalar(0, 0, 255), 2);
            putText(frame, "Face", Point(face.x, face.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
            timings.cpuDrawMs += duration_cast<microseconds>(high_resolution_clock::now() - tDrawStart).count() / 1000.0;

            // --- STAGE 3: CPU EYE DETECTION ---
            auto tEyeStart = high_resolution_clock::now();
            Mat faceROI = smallGray(sFace);
            vector<Rect> eyes;
            eyeCascade.detectMultiScale(faceROI, eyes, 1.1, 4, 0, Size(15, 15));
            timings.cpuEyeMs = duration_cast<microseconds>(high_resolution_clock::now() - tEyeStart).count() / 1000.0;
            totalCpuEyeMs += timings.cpuEyeMs;

            tDrawStart = high_resolution_clock::now();
            for (const Rect &sEye : eyes) {
                Rect eyeGlobal(
                    cvRound((sFace.x + sEye.x) * invScale),
                    cvRound((sFace.y + sEye.y) * invScale),
                    cvRound(sEye.width * invScale),
                    cvRound(sEye.height * invScale)
                );
                // Anatomical eye label from the subject's perspective:
                // image-left is the subject's RIGHT eye; image-right is LEFT eye.
                double eyeCenterX = sEye.x + (sEye.width * 0.5);
                double faceCenterX = sFace.width * 0.5;
                string eyeLabel = (eyeCenterX < faceCenterX) ? "R Eye" : "L Eye";

                rectangle(frame, eyeGlobal, Scalar(0, 255, 0), 2);
                putText(frame, eyeLabel, Point(eyeGlobal.x, eyeGlobal.y - 4), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 255, 0), 1);
            }
            timings.cpuDrawMs += duration_cast<microseconds>(high_resolution_clock::now() - tDrawStart).count() / 1000.0;

            // --- STAGE 4: CPU EAR ESTIMATION ---
            auto tEarStart = high_resolution_clock::now();
            int earWidth  = static_cast<int>(face.width * 0.18);
            int earHeight = static_cast<int>(face.height * 0.35);
            int earY      = face.y + static_cast<int>(face.height * 0.28);

            Rect leftEarRect(max(0, face.x - static_cast<int>(earWidth * 0.6)), earY, earWidth, earHeight);
            Rect rightEarRect(min(frame.cols - earWidth, face.x + face.width - static_cast<int>(earWidth * 0.4)), earY, earWidth, earHeight);
            timings.cpuEarMs = duration_cast<microseconds>(high_resolution_clock::now() - tEarStart).count() / 1000.0;
            totalCpuEarMs += timings.cpuEarMs;

            tDrawStart = high_resolution_clock::now();
            rectangle(frame, leftEarRect, Scalar(255, 0, 0), 2);
            putText(frame, "L Ear", Point(leftEarRect.x, leftEarRect.y - 4), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 0, 0), 1);

            rectangle(frame, rightEarRect, Scalar(255, 255, 0), 2);
            putText(frame, "R Ear", Point(rightEarRect.x, rightEarRect.y - 4), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 0), 1);
            timings.cpuDrawMs += duration_cast<microseconds>(high_resolution_clock::now() - tDrawStart).count() / 1000.0;

            break;
        }

        totalCpuDrawMs += timings.cpuDrawMs;
        timings.totalFrameCpuMs = timings.cpuFaceMs + timings.cpuEyeMs + timings.cpuEarMs + timings.cpuDrawMs;
        totalCpuActiveMs += timings.totalFrameCpuMs;

        // Periodic System Telemetry Check
        auto now = high_resolution_clock::now();
        double intervalSec = duration_cast<milliseconds>(now - lastBenchmarkTime).count() / 1000.0;
        if (intervalSec >= 0.5) {
            CpuSnapshot currCpuSnap = readCpuSnapshot();
            currentCpuPct = calculateCpuUsage(lastCpuSnap, currCpuSnap);
            currentGpuPct = readGpuUsage();
            lastCpuSnap = currCpuSnap;
            lastBenchmarkTime = now;

            // Benchmark telemetry written to file.
            // These fields match the stages actually measured by this program.
            benchmarkLog << fixed << setprecision(1)
                         << "[Benchmark] "
                         << setw(2) << duration_cast<seconds>(now - startTime).count() << "s | "
                         << "CPU:" << setw(4) << currentCpuPct << "% | "
                         << "GPU:" << setw(4) << (currentGpuPct >= 0.0 ? to_string((int)currentGpuPct) + "%" : "N/A") << " | "
                         << setprecision(2)
                         << "GPU-Kernel:" << setw(6) << timings.gpuPreprocMs << "ms | "
                         << "CPU-Face:" << setw(6) << timings.cpuFaceMs << "ms | "
                         << "CPU-Eye:" << setw(6) << timings.cpuEyeMs << "ms | "
                         << "CPU-Ear:" << setw(6) << timings.cpuEarMs << "ms | "
                         << "CPU-Draw:" << setw(6) << timings.cpuDrawMs << "ms | "
                         << "CPU-Total:" << setw(6) << timings.totalFrameCpuMs << "ms\n";
            benchmarkLog.flush();
        }

        double frameDeltaMs = duration_cast<microseconds>(now - lastFrameTimestamp).count() / 1000.0;
        lastFrameTimestamp = now;
        double instantFps = (frameDeltaMs > 0.0) ? (1000.0 / frameDeltaMs) : 0.0;
        double elapsedSec = duration_cast<milliseconds>(now - startTime).count() / 1000.0;

        // Render On-Screen Display HUD
        drawDetailedHUD(frame, currentCpuPct, currentGpuPct, timings, instantFps, frameCount + 1, elapsedSec);

        frameBuffer.push_back(frame.clone());
        frameCount++;
    }

    auto totalTimeMs = duration_cast<milliseconds>(high_resolution_clock::now() - startTime).count();
    if (frameCount == 0 || totalTimeMs == 0) {
        cerr << "No frames captured.\n";
        cleanupOpenCL(ocl);
        return 1;
    }

    double effectiveFps = (frameCount * 1000.0) / totalTimeMs;

    // Final benchmark summary written to benchmark.log.
    // GPU preprocessing below is kernel execution time only, because that is
    // what this program measures with OpenCL event profiling.
    benchmarkLog << "\n=================================== FINAL BENCHMARK SUMMARY ===================================\n";
    benchmarkLog << fixed << setprecision(2);
    benchmarkLog << "Program runtime                    : " << totalTimeMs / 1000.0 << " s\n";
    benchmarkLog << "Observed video throughput          : " << effectiveFps << " FPS\n";
    benchmarkLog << "-----------------------------------------------------------------------------------------------\n";
    benchmarkLog << "AVERAGE PER-FRAME LATENCIES:\n";
    benchmarkLog << "  GPU preprocessing kernel         : " << setw(8) << totalGpuPreprocMs / frameCount << " ms/frame\n";
    benchmarkLog << "  CPU face detection               : " << setw(8) << totalCpuFaceMs / frameCount    << " ms/frame\n";
    benchmarkLog << "  CPU eye detection                : " << setw(8) << totalCpuEyeMs / frameCount     << " ms/frame\n";
    benchmarkLog << "  CPU ear estimation               : " << setw(8) << totalCpuEarMs / frameCount     << " ms/frame\n";
    benchmarkLog << "  CPU drawing/annotation           : " << setw(8) << totalCpuDrawMs / frameCount    << " ms/frame\n";
    benchmarkLog << "  CPU total measured work          : " << setw(8) << totalCpuActiveMs / frameCount  << " ms/frame\n";
    benchmarkLog << "-----------------------------------------------------------------------------------------------\n";
    benchmarkLog << "CUMULATIVE MEASURED STAGE TIMES:\n";
    benchmarkLog << "  GPU preprocessing kernel         : " << setw(8) << totalGpuPreprocMs << " ms\n";
    benchmarkLog << "  CPU face detection               : " << setw(8) << totalCpuFaceMs    << " ms\n";
    benchmarkLog << "  CPU eye detection                : " << setw(8) << totalCpuEyeMs     << " ms\n";
    benchmarkLog << "  CPU ear estimation               : " << setw(8) << totalCpuEarMs     << " ms\n";
    benchmarkLog << "  CPU drawing/annotation           : " << setw(8) << totalCpuDrawMs    << " ms\n";
    benchmarkLog << "  CPU total measured work          : " << setw(8) << totalCpuActiveMs  << " ms\n";
    benchmarkLog << "===============================================================================================\n";
    benchmarkLog.flush();

    cout << "Benchmark complete. Results saved to benchmark.log\n";

    string outputPipeline = "appsrc ! videoconvert ! v4l2h264enc ! "
                           "video/x-h264,profile=baseline ! h264parse ! mp4mux ! "
                           "filesink location=output.mp4";

    VideoWriter writer;
    writer.open(outputPipeline, CAP_GSTREAMER, 0, effectiveFps, frame.size(), true);
    if (!writer.isOpened()) {
        cerr << "Hardware encoder pipeline failed, using software x264enc fallback...\n";
        outputPipeline = "appsrc ! videoconvert ! x264enc tune=zerolatency ! "
                         "video/x-h264,profile=baseline ! h264parse ! mp4mux ! "
                         "filesink location=output.mp4";
        writer.open(outputPipeline, CAP_GSTREAMER, 0, effectiveFps, frame.size(), true);
        if (!writer.isOpened()) {
            cerr << "Could not open fallback video writer.\n";
            cleanupOpenCL(ocl);
            return 1;
        }
    }

    cout << "Encoding video stream to disk with telemetry overlays...\n";
    for (const auto &bufferedFrame : frameBuffer) {
        writer.write(bufferedFrame);
    }

    writer.release();
    cap.release();
    cleanupOpenCL(ocl);

    cout << "Saved output.mp4 (" << totalTimeMs / 1000.0 << "s at " << effectiveFps << " FPS)\n";
    return 0;
}