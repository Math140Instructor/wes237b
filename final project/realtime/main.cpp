#include <arpa/inet.h>
#include <cmath>
#include <iostream>
#include <netinet/in.h>
#include <opencv2/opencv.hpp>
#include <sys/socket.h>
#include <unistd.h>

using namespace cv;
using namespace std;

// Send all bytes to browser
bool sendAll(int socketFd, const void *data, size_t length) {
  const char *buffer = static_cast<const char *>(data);

  while (length > 0) {
    ssize_t sent = send(socketFd, buffer, length, MSG_NOSIGNAL);

    if (sent <= 0) {
      return false;
    }

    buffer += sent;
    length -= sent;
  }

  return true;
}

int main(int argc, char *argv[]) {
  // Default server settings
  string serverIp = "127.0.0.1";
  int port = 5000;

  // Optional:
  // ./main <IP> <port>
  if (argc == 3) {
    serverIp = argv[1];

    try {
      port = stoi(argv[2]);
    } catch (...) {
      cerr << "Invalid port\n";
      return 1;
    }
  } else if (argc != 1) {
    cerr << "Usage: " << argv[0] << " [IP port]\n";
    cerr << "Default: localhost:5000\n";
    cerr << "Example: " << argv[0] << " 0.0.0.0 5000\n";
    return 1;
  }

  if (port < 1 || port > 65535) {
    cerr << "Invalid port\n";
    return 1;
  }

  // RB3 Gen 2 camera input using Qualcomm GStreamer camera source
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

  // Load Haar cascades
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

  // Load transparent glasses PNG
  Mat glasses = imread("../assets/glasses.png", IMREAD_UNCHANGED);

  if (glasses.empty()) {
    cerr << "Could not load glasses.png\n";
    return 1;
  }

  if (glasses.channels() != 4) {
    cerr << "glasses.png must have an alpha channel\n";
    return 1;
  }

  // Read first frame
  Mat frame;

  if (!cap.read(frame) || frame.empty()) {
    cerr << "Could not read first camera frame\n";
    return 1;
  }

  cout << "Camera resolution: " << frame.cols << "x" << frame.rows << endl;

  // ============================================================
  // HTTP server
  // ============================================================

  int serverSocket = socket(AF_INET, SOCK_STREAM, 0);

  if (serverSocket < 0) {
    cerr << "Could not create server socket\n";
    return 1;
  }

  int reuse = 1;

  setsockopt(serverSocket,
             SOL_SOCKET,
             SO_REUSEADDR,
             &reuse,
             sizeof(reuse));

  sockaddr_in serverAddress{};

  serverAddress.sin_family = AF_INET;
  serverAddress.sin_port = htons(port);

  // Allow "localhost" as command-line argument
  if (serverIp == "localhost") {
    serverIp = "127.0.0.1";
  }

  if (serverIp == "0.0.0.0") {
    serverAddress.sin_addr.s_addr = INADDR_ANY;
  } else {
    if (inet_pton(AF_INET,
                  serverIp.c_str(),
                  &serverAddress.sin_addr) != 1) {

      cerr << "Invalid IP address: " << serverIp << "\n";
      close(serverSocket);
      return 1;
    }
  }

  // ============================================================
  // Keep retrying bind() indefinitely
  // ============================================================

  while (bind(serverSocket,
              reinterpret_cast<sockaddr *>(&serverAddress),
              sizeof(serverAddress)) < 0) {

    cerr << "Could not bind to "
         << serverIp << ":"
         << port
         << ". Retrying...\n";

    sleep(1);
  }

  cout << "Successfully bound to "
       << serverIp << ":"
       << port << "\n";

  if (listen(serverSocket, 1) < 0) {
    cerr << "Could not listen on port " << port << "\n";
    close(serverSocket);
    return 1;
  }

  cout << "Video server listening on:\n";

  if (serverIp == "127.0.0.1") {
    cout << "http://localhost:" << port << "\n";
  } else {
    cout << "http://" << serverIp << ":" << port << "\n";
  }

  cout << "Waiting for browser connection...\n";

  // ============================================================
  // Accept clients forever
  // ============================================================

  while (true) {
    sockaddr_in clientAddress{};
    socklen_t clientLength = sizeof(clientAddress);

    int clientSocket =
        accept(serverSocket,
               reinterpret_cast<sockaddr *>(&clientAddress),
               &clientLength);

    if (clientSocket < 0) {
      cerr << "Could not accept browser connection. Retrying...\n";
      continue;
    }

    cout << "Browser connected\n";

    string httpHeader =
        "HTTP/1.1 200 OK\r\n"
        "Cache-Control: no-cache\r\n"
        "Pragma: no-cache\r\n"
        "Connection: close\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
        "\r\n";

    if (!sendAll(clientSocket,
                 httpHeader.data(),
                 httpHeader.size())) {

      cerr << "Could not send HTTP header\n";
      close(clientSocket);
      continue;
    }

    int frameCount = 0;

    // ==========================================================
    // Stream indefinitely until browser disconnects
    // ==========================================================

    while (cap.read(frame) && !frame.empty()) {
      Mat gray;

      cvtColor(frame, gray, COLOR_BGR2GRAY);

      vector<Rect> faces;

      faceCascade.detectMultiScale(gray, faces, 1.1, 5);

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

        // Make eye1 the left-most detected eye
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

        Point2f center((eye1.x + eye2.x) / 2.0f,
                       (eye1.y + eye2.y) / 2.0f);

        // Eye debug markers
        circle(frame, eye1, 5, Scalar(0, 255, 0), -1);
        circle(frame, eye2, 5, Scalar(0, 255, 0), -1);

        // Scale glasses relative to eye distance
        int width = static_cast<int>(eyeDistance * 2.2);

        if (width <= 0) {
          continue;
        }

        int height =
            static_cast<int>(width *
                             static_cast<double>(glasses.rows) /
                             glasses.cols);

        if (height <= 0) {
          continue;
        }

        Mat resized;

        resize(glasses, resized, Size(width, height));

        Point2f glassCenter(resized.cols / 2.0f,
                            resized.rows / 2.0f);

        Mat rotation =
            getRotationMatrix2D(glassCenter, angle, 1.0);

        Mat rotated;

        warpAffine(resized,
                   rotated,
                   rotation,
                   resized.size(),
                   INTER_LINEAR,
                   BORDER_CONSTANT,
                   Scalar(0, 0, 0, 0));

        int x =
            static_cast<int>(center.x - rotated.cols / 2.0f);

        int y =
            static_cast<int>(center.y - rotated.rows / 2.0f);

        // Alpha blend glasses onto camera frame
        for (int gy = 0; gy < rotated.rows; gy++) {

          for (int gx = 0; gx < rotated.cols; gx++) {

            int fx = x + gx;
            int fy = y + gy;

            if (fx < 0 || fy < 0 ||
                fx >= frame.cols ||
                fy >= frame.rows) {
              continue;
            }

            Vec4b pixel = rotated.at<Vec4b>(gy, gx);

            float alpha = pixel[3] / 255.0f;

            for (int c = 0; c < 3; c++) {

              frame.at<Vec3b>(fy, fx)[c] =
                  static_cast<uchar>(
                      pixel[c] * alpha +
                      frame.at<Vec3b>(fy, fx)[c] *
                          (1.0f - alpha));
            }
          }
        }

        // Only process first valid detected face
        break;
      }

      // ========================================================
      // Send processed frame over HTTP
      // ========================================================

      vector<uchar> jpeg;

      imencode(".jpg", frame, jpeg);

      string frameHeader =
          "--frame\r\n"
          "Content-Type: image/jpeg\r\n"
          "Content-Length: " +
          to_string(jpeg.size()) +
          "\r\n\r\n";

      if (!sendAll(clientSocket,
                   frameHeader.data(),
                   frameHeader.size())) {
        cerr << "Browser disconnected\n";
        break;
      }

      if (!sendAll(clientSocket,
                   jpeg.data(),
                   jpeg.size())) {
        cerr << "Browser disconnected\n";
        break;
      }

      const char *frameEnd = "\r\n";

      if (!sendAll(clientSocket, frameEnd, 2)) {
        cerr << "Browser disconnected\n";
        break;
      }

      frameCount++;

      if (frameCount % 30 == 0) {
        cout << "Processed " << frameCount << " frames" << endl;
      }
    }

    close(clientSocket);

    cout << "Waiting for next browser connection...\n";
  }

  close(serverSocket);
  cap.release();

  return 0;
}