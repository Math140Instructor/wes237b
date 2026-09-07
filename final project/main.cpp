#include <arpa/inet.h>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <iostream>
#include <mutex>
#include <netinet/in.h>
#include <opencv2/opencv.hpp>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace cv;
using namespace std;

// ============================================================
// Shared JPEG frame
// ============================================================

mutex frameMutex;
condition_variable frameCondition;

vector<uchar> latestJpeg;

uint64_t frameSequence = 0;

atomic<bool> running(true);

// ============================================================
// Send entire buffer over socket
// ============================================================

bool sendAll(int socketFd, const void *data, size_t size) {
  const char *ptr = static_cast<const char *>(data);

  size_t sentTotal = 0;

  while (sentTotal < size) {

    ssize_t sent = send(socketFd, ptr + sentTotal, size - sentTotal, MSG_NOSIGNAL);

    if (sent <= 0) {
      return false;
    }

    sentTotal += sent;
  }

  return true;
}

// ============================================================
// Handle one browser connection
// ============================================================

void handleClient(int clientSocket) {

  char requestBuffer[4096];

  ssize_t bytesReceived = recv(clientSocket, requestBuffer, sizeof(requestBuffer) - 1, 0);

  if (bytesReceived <= 0) {
    close(clientSocket);
    return;
  }

  requestBuffer[bytesReceived] = '\0';

  string request(requestBuffer);

  // ----------------------------------------------------------
  // MJPEG stream
  // ----------------------------------------------------------

  if (request.find("GET /stream") != string::npos) {

    string header = "HTTP/1.1 200 OK\r\n"
                    "Cache-Control: no-cache, no-store, must-revalidate\r\n"
                    "Pragma: no-cache\r\n"
                    "Expires: 0\r\n"
                    "Connection: close\r\n"
                    "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
                    "\r\n";

    if (!sendAll(clientSocket, header.data(), header.size())) {
      close(clientSocket);
      return;
    }

    uint64_t lastSequence = 0;

    while (running) {

      vector<uchar> jpeg;

      {
        unique_lock<mutex> lock(frameMutex);

        frameCondition.wait(lock, [&]() { return !running || frameSequence != lastSequence; });

        if (!running) {
          break;
        }

        jpeg = latestJpeg;

        lastSequence = frameSequence;
      }

      if (jpeg.empty()) {
        continue;
      }

      string frameHeader = "--frame\r\n"
                           "Content-Type: image/jpeg\r\n"
                           "Content-Length: " +
                           to_string(jpeg.size()) + "\r\n\r\n";

      if (!sendAll(clientSocket, frameHeader.data(), frameHeader.size())) {
        break;
      }

      if (!sendAll(clientSocket, jpeg.data(), jpeg.size())) {
        break;
      }

      const char *frameEnd = "\r\n";

      if (!sendAll(clientSocket, frameEnd, 2)) {
        break;
      }
    }

    close(clientSocket);
    return;
  }

  // ----------------------------------------------------------
  // Main web page
  // ----------------------------------------------------------

  string html = "<!DOCTYPE html>"
                "<html>"
                "<head>"
                "<title>RB3 Camera</title>"
                "<meta name=\"viewport\" "
                "content=\"width=device-width, initial-scale=1\">"
                "<style>"
                "body {"
                "  background:#111;"
                "  color:white;"
                "  text-align:center;"
                "  font-family:Arial,sans-serif;"
                "  margin:0;"
                "  padding:20px;"
                "}"
                "img {"
                "  max-width:100%;"
                "  height:auto;"
                "}"
                "</style>"
                "</head>"
                "<body>"
                "<h1>RB3 Gen 2 Camera</h1>"
                "<img src=\"/stream\">"
                "</body>"
                "</html>";

  string response = "HTTP/1.1 200 OK\r\n"
                    "Content-Type: text/html\r\n"
                    "Content-Length: " +
                    to_string(html.size()) +
                    "\r\n"
                    "Connection: close\r\n"
                    "\r\n" +
                    html;

  sendAll(clientSocket, response.data(), response.size());

  close(clientSocket);
}

// ============================================================
// HTTP server
// ============================================================

void webServer(int port) {

  int serverSocket = socket(AF_INET, SOCK_STREAM, 0);

  if (serverSocket < 0) {
    cerr << "Could not create HTTP socket\n";
    running = false;
    return;
  }

  int enable = 1;

  setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable));

  sockaddr_in address{};

  address.sin_family = AF_INET;

  // Listen on all interfaces
  address.sin_addr.s_addr = INADDR_ANY;

  address.sin_port = htons(port);

  if (bind(serverSocket, reinterpret_cast<sockaddr *>(&address), sizeof(address)) < 0) {

    cerr << "Could not bind HTTP server to port " << port << "\n";

    close(serverSocket);

    running = false;

    return;
  }

  if (listen(serverSocket, 10) < 0) {

    cerr << "Could not listen on HTTP socket\n";

    close(serverSocket);

    running = false;

    return;
  }

  cout << "\nWeb server started\n";
  cout << "Open: http://<RB3-IP>:" << port << "\n\n";

  while (running) {

    sockaddr_in clientAddress{};

    socklen_t clientLength = sizeof(clientAddress);

    int clientSocket = accept(serverSocket, reinterpret_cast<sockaddr *>(&clientAddress), &clientLength);

    if (clientSocket < 0) {

      if (running) {
        cerr << "HTTP accept failed\n";
      }

      continue;
    }

    // One lightweight thread per browser connection
    thread(handleClient, clientSocket).detach();
  }

  close(serverSocket);
}

// ============================================================
// Main
// ============================================================

int main() {

  // Prevent disconnected browser sockets from terminating program
  signal(SIGPIPE, SIG_IGN);

  // ----------------------------------------------------------
  // RB3 camera
  // ----------------------------------------------------------

  string inputPipeline = "qtiqmmfsrc camera=0 ! "
                         "video/x-raw,format=NV12,width=640,height=480,framerate=30/1 ! "
                         "videoconvert ! "
                         "video/x-raw,format=BGR ! "
                         "appsink max-buffers=1 drop=true sync=false";

  VideoCapture cap(inputPipeline, CAP_GSTREAMER);

  if (!cap.isOpened()) {

    cerr << "Could not open RB3 camera\n";

    return 1;
  }

  cout << "RB3 camera opened\n";

  // ----------------------------------------------------------
  // Haar cascades
  // ----------------------------------------------------------

  CascadeClassifier faceCascade;
  CascadeClassifier eyeCascade;

  if (!faceCascade.load("/usr/share/opencv4/haarcascades/"
                        "haarcascade_frontalface_default.xml")) {

    cerr << "Could not load face cascade\n";

    return 1;
  }

  if (!eyeCascade.load("/usr/share/opencv4/haarcascades/"
                       "haarcascade_eye.xml")) {

    cerr << "Could not load eye cascade\n";

    return 1;
  }

  // ----------------------------------------------------------
  // Glasses image
  // ----------------------------------------------------------

  Mat glasses = imread("glasses.png", IMREAD_UNCHANGED);

  if (glasses.empty()) {

    cerr << "Could not load glasses.png\n";

    return 1;
  }

  if (glasses.channels() != 4) {

    cerr << "glasses.png must have alpha channel\n";

    return 1;
  }

  // ----------------------------------------------------------
  // Start web server
  // ----------------------------------------------------------

  thread serverThread(webServer, 8080);

  // ----------------------------------------------------------
  // Camera processing loop
  // ----------------------------------------------------------

  Mat frame;

  while (running) {

    if (!cap.read(frame) || frame.empty()) {

      cerr << "Could not read camera frame\n";

      break;
    }

    Mat gray;

    cvtColor(frame, gray, COLOR_BGR2GRAY);

    vector<Rect> faces;

    faceCascade.detectMultiScale(gray, faces, 1.1, 5);

    // ========================================================
    // Face processing
    // ========================================================

    for (const Rect &face : faces) {

      Mat faceROI = gray(face);

      vector<Rect> eyes;

      eyeCascade.detectMultiScale(faceROI, eyes, 1.1, 5);

      if (eyes.size() < 2) {
        continue;
      }

      Point2f eye1(face.x + eyes[0].x + eyes[0].width / 2.0f,

                   face.y + eyes[0].y + eyes[0].height / 2.0f);

      Point2f eye2(face.x + eyes[1].x + eyes[1].width / 2.0f,

                   face.y + eyes[1].y + eyes[1].height / 2.0f);

      if (eye1.x > eye2.x) {
        swap(eye1, eye2);
      }

      double dx = eye2.x - eye1.x;

      double dy = eye2.y - eye1.y;

      double angle = atan2(dy, dx) * 180.0 / CV_PI;

      double eyeDistance = sqrt(dx * dx + dy * dy);

      if (eyeDistance <= 1.0) {
        continue;
      }

      Point2f center((eye1.x + eye2.x) / 2.0f, (eye1.y + eye2.y) / 2.0f);

      // Debug eye markers
      circle(frame, eye1, 5, Scalar(0, 255, 0), -1);

      circle(frame, eye2, 5, Scalar(0, 255, 0), -1);

      // ------------------------------------------------------
      // Resize glasses
      // ------------------------------------------------------

      int width = static_cast<int>(eyeDistance * 2.2);

      if (width <= 0) {
        continue;
      }

      int height = static_cast<int>(width * static_cast<double>(glasses.rows) / glasses.cols);

      if (height <= 0) {
        continue;
      }

      Mat resized;

      resize(glasses, resized, Size(width, height));

      // ------------------------------------------------------
      // Rotate glasses
      // ------------------------------------------------------

      Point2f glassCenter(resized.cols / 2.0f, resized.rows / 2.0f);

      Mat rotation = getRotationMatrix2D(glassCenter, angle, 1.0);

      Mat rotated;

      warpAffine(resized, rotated, rotation, resized.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0, 0));

      int x = static_cast<int>(center.x - rotated.cols / 2.0f);

      int y = static_cast<int>(center.y - rotated.rows / 2.0f);

      // ------------------------------------------------------
      // Alpha blending
      // ------------------------------------------------------

      for (int gy = 0; gy < rotated.rows; gy++) {

        for (int gx = 0; gx < rotated.cols; gx++) {

          int fx = x + gx;
          int fy = y + gy;

          if (fx < 0 || fy < 0 || fx >= frame.cols || fy >= frame.rows) {

            continue;
          }

          Vec4b pixel = rotated.at<Vec4b>(gy, gx);

          float alpha = pixel[3] / 255.0f;

          for (int c = 0; c < 3; c++) {

            frame.at<Vec3b>(fy, fx)[c] = static_cast<uchar>(pixel[c] * alpha +

                                                            frame.at<Vec3b>(fy, fx)[c] * (1.0f - alpha));
          }
        }
      }

      break;
    }

    // ========================================================
    // JPEG encode for browser
    // ========================================================

    vector<uchar> jpeg;

    vector<int> jpegParameters = {IMWRITE_JPEG_QUALITY, 80};

    if (!imencode(".jpg", frame, jpeg, jpegParameters)) {

      cerr << "JPEG encode failed\n";
      continue;
    }

    // Publish latest frame
    {
      lock_guard<mutex> lock(frameMutex);

      latestJpeg.swap(jpeg);

      frameSequence++;
    }

    frameCondition.notify_all();
  }

  running = false;

  frameCondition.notify_all();

  cap.release();

  if (serverThread.joinable()) {

    // accept() may still be blocked when shutting down.
    // Normally Ctrl+C/process termination handles this.
    serverThread.detach();
  }

  return 0;
}