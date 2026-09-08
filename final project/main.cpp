#include <arpa/inet.h>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace cv;
using namespace std;
using namespace std::chrono;

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

    // Set VideoWriter to realistic processing FPS (e.g. 15.0 FPS) or tune dynamically
    double targetFps = 15.0;

    string outputPipeline = "appsrc ! videoconvert ! x264enc tune=zerolatency ! "
                           "video/x-h264,profile=baseline ! h264parse ! mp4mux ! "
                           "filesink location=output.mp4";

    VideoWriter writer;
    writer.open(outputPipeline, CAP_GSTREAMER, 0, targetFps, frame.size(), true);
    if (!writer.isOpened()) {
        cerr << "Could not open output video pipeline\n";
        return 1;
    }

    int frameCount = 0;
    const int maxFrames = 150; // 150 frames @ ~15 FPS = ~10 seconds
    auto startTime = high_resolution_clock::now();

    do {
        auto frameStart = high_resolution_clock::now();

        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        vector<Rect> faces;
        faceCascade.detectMultiScale(gray, faces, 1.1, 5);

        for (const Rect &face : faces) {
            // 1. Draw Face Box (Red)
            rectangle(frame, face, Scalar(0, 0, 255), 2);
            putText(frame, "Face", Point(face.x, face.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);

            Mat faceROI = gray(face);
            vector<Rect> eyes;
            eyeCascade.detectMultiScale(faceROI, eyes, 1.1, 5);

            // 2. Draw Eye Boxes (Green)
            for (const Rect &eye : eyes) {
                Rect eyeGlobal = eye;
                eyeGlobal.x += face.x;
                eyeGlobal.y += face.y;
                rectangle(frame, eyeGlobal, Scalar(0, 255, 0), 2);
                putText(frame, "Eye", Point(eyeGlobal.x, eyeGlobal.y - 4), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 255, 0), 1);
            }

            // 3. Ear Bounding Box Estimation (Blue & Cyan)
            int earWidth  = face.width * 0.18;
            int earHeight = face.height * 0.35;
            int earY      = face.y + (face.height * 0.28);

            Rect leftEarRect(max(0, face.x - (int)(earWidth * 0.6)), earY, earWidth, earHeight);
            Rect rightEarRect(min(frame.cols - earWidth, face.x + face.width - (int)(earWidth * 0.4)), earY, earWidth, earHeight);

            rectangle(frame, leftEarRect, Scalar(255, 0, 0), 2);
            putText(frame, "L Ear", Point(leftEarRect.x, leftEarRect.y - 4), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 0, 0), 1);

            rectangle(frame, rightEarRect, Scalar(255, 255, 0), 2);
            putText(frame, "R Ear", Point(rightEarRect.x, rightEarRect.y - 4), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 0), 1);

            break;
        }

        writer.write(frame);
        frameCount++;

        if (frameCount % 30 == 0) {
            cout << "Processed " << frameCount << " frames" << endl;
        }

    } while (frameCount < maxFrames && cap.read(frame) && !frame.empty());

    auto totalTime = duration_cast<milliseconds>(high_resolution_clock::now() - startTime).count();
    double effectiveFps = (frameCount * 1000.0) / totalTime;

    cout << "Finished in " << totalTime / 1000.0 << "s (" << effectiveFps << " effective FPS)\n";

    writer.release();
    cap.release();
    cout << "Saved output.mp4\n";
    return 0;
}