#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

int main()
{
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Could not open camera\n";
        return 1;
    }

    CascadeClassifier faceCascade;
    CascadeClassifier eyeCascade;

    if (!faceCascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml") ||
        !eyeCascade.load("/usr/share/opencv4/haarcascades/haarcascade_eye.xml")) {
        cerr << "Could not load cascades\n";
        return 1;
    }

    Mat glasses = imread("glasses.png", IMREAD_UNCHANGED);

    if (glasses.empty()) {
        cerr << "Could not load glasses.png\n";
        return 1;
    }

    Mat frame;

    while (cap.read(frame))
    {
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        vector<Rect> faces;
        faceCascade.detectMultiScale(gray, faces, 1.1, 5);

        for (const Rect& face : faces)
        {
            Mat faceROI = gray(face);

            vector<Rect> eyes;
            eyeCascade.detectMultiScale(faceROI, eyes, 1.1, 5);

            if (eyes.size() < 2)
                continue;

            Point2f eye1(
                face.x + eyes[0].x + eyes[0].width / 2.0f,
                face.y + eyes[0].y + eyes[0].height / 2.0f
            );

            Point2f eye2(
                face.x + eyes[1].x + eyes[1].width / 2.0f,
                face.y + eyes[1].y + eyes[1].height / 2.0f
            );

            // Make sure eye1 = left eye
            if (eye1.x > eye2.x)
                swap(eye1, eye2);

            double dx = eye2.x - eye1.x;
            double dy = eye2.y - eye1.y;

            double angle = atan2(dy, dx) * 180.0 / CV_PI;
            double eyeDistance = sqrt(dx * dx + dy * dy);

            Point2f center(
                (eye1.x + eye2.x) / 2.0f,
                (eye1.y + eye2.y) / 2.0f
            );

            // Draw eyes so you can verify detection
            circle(frame, eye1, 5, Scalar(0,255,0), -1);
            circle(frame, eye2, 5, Scalar(0,255,0), -1);

            // Glasses width relative to eye distance
            int width = static_cast<int>(eyeDistance * 2.2);
            int height = static_cast<int>(
                width * ((double)glasses.rows / glasses.cols)
            );

            Mat resized;
            resize(glasses, resized, Size(width, height));

            Point2f glassCenter(
                resized.cols / 2.0f,
                resized.rows / 2.0f
            );

            Mat rotation =
                getRotationMatrix2D(glassCenter, angle, 1.0);

            Mat rotated;

            warpAffine(
                resized,
                rotated,
                rotation,
                resized.size(),
                INTER_LINEAR,
                BORDER_CONSTANT,
                Scalar(0,0,0,0)
            );

            int x = static_cast<int>(center.x - rotated.cols / 2);
            int y = static_cast<int>(center.y - rotated.rows / 2);

            // Overlay transparent PNG
            for (int gy = 0; gy < rotated.rows; gy++)
            {
                for (int gx = 0; gx < rotated.cols; gx++)
                {
                    int fx = x + gx;
                    int fy = y + gy;

                    if (fx < 0 || fy < 0 ||
                        fx >= frame.cols || fy >= frame.rows)
                        continue;

                    Vec4b pixel = rotated.at<Vec4b>(gy, gx);

                    float alpha = pixel[3] / 255.0f;

                    for (int c = 0; c < 3; c++)
                    {
                        frame.at<Vec3b>(fy, fx)[c] =
                            pixel[c] * alpha +
                            frame.at<Vec3b>(fy, fx)[c] * (1.0f - alpha);
                    }
                }
            }

            // Only process first detected face for quick test
            break;
        }

        imshow("RB3 Face Glasses", frame);

        if (waitKey(1) == 27)
            break;
    }

    return 0;
}