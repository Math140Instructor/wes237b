#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

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

    if (!faceCascade.load(cascadePath + "haarcascade_frontalface_default.xml")) {
        cerr << "Could not load face cascade\n";
        return 1;
    }
    if (!eyeCascade.load(cascadePath + "haarcascade_eye.xml")) {
        cerr << "Could not load eye cascade\n";
        return 1;
    }

    Mat frame;
    if (!cap.read(frame) || frame.empty()) {
        cerr << "Could not read first camera frame\n";
        return 1;
    }

    string outputPipeline = "appsrc ! videoconvert ! x264enc tune=zerolatency ! "
                           "video/x-h264,profile=baseline ! h264parse ! mp4mux ! "
                           "filesink location=output.mp4";

    VideoWriter writer;
    writer.open(outputPipeline, CAP_GSTREAMER, 0, 30.0, frame.size(), true);
    if (!writer.isOpened()) {
        cerr << "Could not open output video pipeline\n";
        return 1;
    }

    int frameCount = 0;
    const int maxFrames = 300;

    do {
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
            // Ear dimensions proportional to face size
            int earWidth  = face.width * 0.18;
            int earHeight = face.height * 0.35;
            int earY      = face.y + (face.height * 0.28); // level with eyes/temple

            // Left Ear (subject's right side)
            Rect leftEarRect(max(0, face.x - (int)(earWidth * 0.6)), earY, earWidth, earHeight);
            // Right Ear (subject's left side)
            Rect rightEarRect(min(frame.cols - earWidth, face.x + face.width - (int)(earWidth * 0.4)), earY, earWidth, earHeight);

            rectangle(frame, leftEarRect, Scalar(255, 0, 0), 2);
            putText(frame, "L Ear", Point(leftEarRect.x, leftEarRect.y - 4), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 0, 0), 1);

            rectangle(frame, rightEarRect, Scalar(255, 255, 0), 2);
            putText(frame, "R Ear", Point(rightEarRect.x, rightEarRect.y - 4), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 0), 1);

            break; // Process primary face
        }

        writer.write(frame);
        frameCount++;

        if (frameCount % 30 == 0) {
            cout << "Processed " << frameCount << " frames" << endl;
        }

    } while (frameCount < maxFrames && cap.read(frame) && !frame.empty());

    writer.release();
    cap.release();
    cout << "Saved output.mp4\n";
    return 0;
}