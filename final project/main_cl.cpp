#include <cmath>
#include <iostream>
#include <opencv2/core/ocl.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// -----------------------------------------------------------------------------
// OpenCL alpha blending using OpenCV UMat
//
// frame:
//      BGR camera image
//
// overlay:
//      BGRA rotated glasses
//
// x, y:
//      upper-left position of overlay in frame
// -----------------------------------------------------------------------------
void alphaBlendOpenCL(UMat &frame, const UMat &overlay, int x, int y) {
  // ------------------------------------------------------------
  // Clip overlay against frame boundaries
  // ------------------------------------------------------------

  int frameX = max(0, x);
  int frameY = max(0, y);

  int overlayX = max(0, -x);
  int overlayY = max(0, -y);

  int width = min(overlay.cols - overlayX, frame.cols - frameX);

  int height = min(overlay.rows - overlayY, frame.rows - frameY);

  if (width <= 0 || height <= 0)
    return;

  Rect frameRect(frameX, frameY, width, height);

  Rect overlayRect(overlayX, overlayY, width, height);

  UMat frameROI = frame(frameRect);
  UMat overlayROI = overlay(overlayRect);

  // ------------------------------------------------------------
  // Split BGRA glasses into channels
  // ------------------------------------------------------------

  vector<UMat> overlayChannels;

  split(overlayROI, overlayChannels);

  if (overlayChannels.size() != 4)
    return;

  // ------------------------------------------------------------
  // Create BGR glasses image
  // ------------------------------------------------------------

  vector<UMat> bgrChannels = {overlayChannels[0], overlayChannels[1], overlayChannels[2]};

  UMat overlayBGR;

  merge(bgrChannels, overlayBGR);

  // ------------------------------------------------------------
  // Convert alpha:
  //
  //      0..255
  //
  // into:
  //
  //      0.0..1.0
  // ------------------------------------------------------------

  UMat alpha;

  overlayChannels[3].convertTo(alpha, CV_32F, 1.0 / 255.0);

  // Need three alpha channels for BGR multiplication

  vector<UMat> alphaChannels = {alpha, alpha, alpha};

  UMat alpha3;

  merge(alphaChannels, alpha3);

  // ------------------------------------------------------------
  // Convert images to floating point
  // ------------------------------------------------------------

  UMat overlayFloat;
  UMat frameFloat;

  overlayBGR.convertTo(overlayFloat, CV_32FC3);

  frameROI.convertTo(frameFloat, CV_32FC3);

  // ------------------------------------------------------------
  // result =
  //
  // glasses * alpha
  // +
  // frame * (1 - alpha)
  // ------------------------------------------------------------

  UMat foreground;
  UMat background;
  UMat inverseAlpha;
  UMat resultFloat;

  multiply(overlayFloat, alpha3, foreground);

  subtract(Scalar::all(1.0), alpha3, inverseAlpha);

  multiply(frameFloat, inverseAlpha, background);

  add(foreground, background, resultFloat);

  // ------------------------------------------------------------
  // Convert back to normal 8-bit BGR image
  // ------------------------------------------------------------

  resultFloat.convertTo(frameROI, CV_8UC3);
}

// -----------------------------------------------------------------------------
// MAIN
// -----------------------------------------------------------------------------

int main() {
  // =========================================================================
  // ENABLE OPENCL
  // =========================================================================

  cout << "Checking OpenCL..." << endl;

  if (!ocl::haveOpenCL()) {
    cerr << "OpenCV reports that OpenCL is not available." << endl;
    return 1;
  }

  ocl::setUseOpenCL(true);

  if (!ocl::useOpenCL()) {
    cerr << "OpenCL exists but OpenCV could not enable it." << endl;
    return 1;
  }

  cout << "OpenCL ENABLED" << endl;

  // Display GPU/device information

  ocl::Device device = ocl::Device::getDefault();

  cout << "OpenCL device: " << device.name() << endl;
  cout << "OpenCL vendor: " << device.vendorName() << endl;
  cout << "OpenCL version: " << device.OpenCLVersion() << endl;
  cout << "OpenCV OpenCL available: " << (cv::ocl::haveOpenCL() ? "YES" : "NO") << endl;

  // =========================================================================
  // RB3 CAMERA
  // =========================================================================

  string inputPipeline = "qtiqmmfsrc camera=0 ! "
                         "video/x-raw,format=NV12,width=1280,height=720,framerate=30/1 ! "
                         "videoconvert ! "
                         "video/x-raw,format=BGR ! "
                         "appsink drop=true sync=false";

  VideoCapture cap(inputPipeline, CAP_GSTREAMER);

  if (!cap.isOpened()) {
    cerr << "Could not open RB3 camera" << endl;
    return 1;
  }

  cout << "RB3 camera opened" << endl;

  // =========================================================================
  // HAAR CASCADES
  // =========================================================================

  CascadeClassifier faceCascade;
  CascadeClassifier eyeCascade;

  if (!faceCascade.load("/usr/share/opencv4/haarcascades/"
                        "haarcascade_frontalface_default.xml")) {
    cerr << "Could not load face cascade" << endl;
    return 1;
  }

  if (!eyeCascade.load("/usr/share/opencv4/haarcascades/"
                       "haarcascade_eye.xml")) {
    cerr << "Could not load eye cascade" << endl;
    return 1;
  }

  // =========================================================================
  // LOAD GLASSES
  // =========================================================================

  Mat glassesCPU = imread("glasses.png", IMREAD_UNCHANGED);

  if (glassesCPU.empty()) {
    cerr << "Could not load glasses.png" << endl;
    return 1;
  }

  if (glassesCPU.channels() != 4) {
    cerr << "glasses.png must contain an alpha channel" << endl;
    return 1;
  }

  // Upload glasses to OpenCL memory

  UMat glasses;

  glassesCPU.copyTo(glasses);

  // =========================================================================
  // FIRST FRAME
  // =========================================================================

  Mat frameCPU;

  if (!cap.read(frameCPU) || frameCPU.empty()) {
    cerr << "Could not read first camera frame" << endl;
    return 1;
  }

  cout << "Camera resolution: " << frameCPU.cols << "x" << frameCPU.rows << endl;

  // =========================================================================
  // OUTPUT VIDEO
  // =========================================================================

  string outputPipeline = "appsrc ! "
                          "videoconvert ! "
                          "x264enc tune=zerolatency ! "
                          "video/x-h264,profile=baseline ! "
                          "h264parse ! "
                          "mp4mux ! "
                          "filesink location=output.mp4";

  VideoWriter writer;

  writer.open(outputPipeline, CAP_GSTREAMER, 0, 30.0, frameCPU.size(), true);

  if (!writer.isOpened()) {
    cerr << "Could not open output video pipeline" << endl;
    return 1;
  }

  cout << "Recording processed video to output.mp4" << endl;

  // =========================================================================
  // PROCESS VIDEO
  // =========================================================================

  int frameCount = 0;

  const int maxFrames = 300;

  do {
    // ---------------------------------------------------------------------
    // Upload camera frame into OpenCL memory
    // ---------------------------------------------------------------------

    UMat frame;

    frameCPU.copyTo(frame);

    // ---------------------------------------------------------------------
    // OPENCL:
    //
    // BGR -> grayscale
    // ---------------------------------------------------------------------

    UMat gray;

    cvtColor(frame, gray, COLOR_BGR2GRAY);

    // ---------------------------------------------------------------------
    // Haar cascade detection
    //
    // CascadeClassifier is still essentially CPU-side.
    //
    // getMat() maps/downloads the grayscale UMat for the detector.
    // ---------------------------------------------------------------------

    Mat grayCPU = gray.getMat(ACCESS_READ);

    vector<Rect> faces;

    faceCascade.detectMultiScale(grayCPU, faces, 1.1, 5);

    // ---------------------------------------------------------------------
    // Faces
    // ---------------------------------------------------------------------

    for (const Rect &face : faces) {
      Mat faceROI = grayCPU(face);

      vector<Rect> eyes;

      eyeCascade.detectMultiScale(faceROI, eyes, 1.1, 5);

      if (eyes.size() < 2)
        continue;

      // -----------------------------------------------------------------
      // Eye centers
      // -----------------------------------------------------------------

      Point2f eye1(face.x + eyes[0].x + eyes[0].width / 2.0f,

                   face.y + eyes[0].y + eyes[0].height / 2.0f);

      Point2f eye2(face.x + eyes[1].x + eyes[1].width / 2.0f,

                   face.y + eyes[1].y + eyes[1].height / 2.0f);

      // Make eye1 the left-most eye

      if (eye1.x > eye2.x) {
        swap(eye1, eye2);
      }

      // -----------------------------------------------------------------
      // Eye angle and distance
      // -----------------------------------------------------------------

      double dx = eye2.x - eye1.x;

      double dy = eye2.y - eye1.y;

      double angle = atan2(dy, dx) * 180.0 / CV_PI;

      double eyeDistance = sqrt(dx * dx + dy * dy);

      if (eyeDistance <= 1.0)
        continue;

      Point2f center((eye1.x + eye2.x) / 2.0f, (eye1.y + eye2.y) / 2.0f);

      // -----------------------------------------------------------------
      // OPENCL:
      //
      // Draw eye markers
      // -----------------------------------------------------------------

      circle(frame, eye1, 5, Scalar(0, 255, 0), -1);

      circle(frame, eye2, 5, Scalar(0, 255, 0), -1);

      // -----------------------------------------------------------------
      // Glasses dimensions
      // -----------------------------------------------------------------

      int width = static_cast<int>(eyeDistance * 2.2);

      if (width <= 0)
        continue;

      int height = static_cast<int>(width * static_cast<double>(glasses.rows) / glasses.cols);

      if (height <= 0)
        continue;

      // -----------------------------------------------------------------
      // OPENCL:
      //
      // Resize glasses
      // -----------------------------------------------------------------

      UMat resized;

      resize(glasses, resized, Size(width, height), 0.0, 0.0, INTER_LINEAR);

      // -----------------------------------------------------------------
      // Rotation matrix
      // -----------------------------------------------------------------

      Point2f glassCenter(resized.cols / 2.0f, resized.rows / 2.0f);

      Mat rotation = getRotationMatrix2D(glassCenter, angle, 1.0);

      // -----------------------------------------------------------------
      // OPENCL:
      //
      // Rotate glasses
      // -----------------------------------------------------------------

      UMat rotated;

      warpAffine(resized, rotated, rotation, resized.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0, 0));

      // -----------------------------------------------------------------
      // Glasses position
      // -----------------------------------------------------------------

      int x = static_cast<int>(center.x - rotated.cols / 2.0f);

      int y = static_cast<int>(center.y - rotated.rows / 2.0f);

      // -----------------------------------------------------------------
      // OPENCL:
      //
      // Alpha blend glasses onto frame
      //
      // This replaces your nested CPU gy/gx loop.
      // -----------------------------------------------------------------

      alphaBlendOpenCL(frame, rotated, x, y);

      // Only first valid face

      break;
    }

    // ---------------------------------------------------------------------
    // Bring processed image back from OpenCL/GPU
    // ---------------------------------------------------------------------

    Mat outputFrame = frame.getMat(ACCESS_READ);

    // ---------------------------------------------------------------------
    // Write video
    // ---------------------------------------------------------------------

    writer.write(outputFrame);

    frameCount++;

    if (frameCount % 30 == 0) {
      cout << "Processed " << frameCount << " frames" << endl;
    }

  } while (frameCount < maxFrames && cap.read(frameCPU) && !frameCPU.empty());

  // =========================================================================
  // CLEANUP
  // =========================================================================

  writer.release();

  cap.release();

  cout << "Saved output.mp4" << endl;

  return 0;
}