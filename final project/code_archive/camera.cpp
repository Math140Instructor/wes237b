#include <cstdlib>
#include <iostream>
#include <string>

using namespace std;

int main(int argc, char **argv) {

  string host = "127.0.0.1";
  int port = 5000;

  if (argc >= 2)
    host = argv[1];

  if (argc >= 3)
    port = stoi(argv[2]);

  // ==========================================================
  // Latency-first video settings
  // ==========================================================

  const int width = 640;
  const int height = 480;
  const int fps = 30;

  // All-I-frame video needs more bitrate than normal H.264.
  const int bitrateKbps = 3000;

  // ~1 frame at 30 FPS
  const int fragmentMs = 33;

  cout << "RB3 low-latency camera sender\n";
  cout << "Video: "
       << width << "x"
       << height << " @ "
       << fps << " FPS\n";

  cout << "Destination: "
       << host << ":"
       << port << "\n";

  // ==========================================================
  // Camera -> H264 -> fragmented MP4 -> TCP
  //
  // key-int-max=1:
  // Every frame is independently decodable.
  //
  // This is intentional. It allows stale frames to be dropped
  // by the web client without breaking H264 dependencies.
  // ==========================================================

  string pipeline =
      "gst-launch-1.0 -e "

      "qtiqmmfsrc camera=0 ! "

      "'video/x-raw,"
      "format=NV12,"
      "width=" +
      to_string(width) +
      ",height=" +
      to_string(height) +
      ",framerate=" +
      to_string(fps) +
      "/1' ! "

      "videoconvert ! "

      "'video/x-raw,format=I420' ! "

      // ------------------------------------------------------
      // Extremely low-latency H264
      // ------------------------------------------------------

      "x264enc "

      "tune=zerolatency "
      "speed-preset=ultrafast "

      "bitrate=" +
      to_string(bitrateKbps) +
      " "

      // Every frame is a keyframe.
      "key-int-max=1 "

      // No future-frame dependencies.
      "bframes=0 "

      // Minimum reference-frame buffering.
      "ref=1 "

      // Disable rate-control lookahead.
      "rc-lookahead=0 "

      // Disable threaded lookahead.
      "sync-lookahead=0 "

      // Lower-latency x264 threading.
      "sliced-threads=true "

      // Small VBV buffer.
      "vbv-buf-capacity=50 "

      "byte-stream=true ! "

      // ------------------------------------------------------
      // Convert H264 to AVC for fragmented MP4
      // ------------------------------------------------------

      "h264parse "
      "config-interval=-1 ! "

      "'video/x-h264,"
      "stream-format=avc,"
      "alignment=au' ! "

      // ------------------------------------------------------
      // Fragmented MP4
      //
      // Approximately one fragment per frame.
      // ------------------------------------------------------

      "mp4mux "
      "fragment-duration=" +
      to_string(fragmentMs) +
      " "
      "streamable=true ! "

      // ------------------------------------------------------
      // Send directly to web.cpp
      // ------------------------------------------------------

      "tcpclientsink "
      "host=" +
      host +
      " "
      "port=" +
      to_string(port) +
      " "
      "sync=false "
      "async=false";

  cout << "\nStarting pipeline:\n\n";
  cout << pipeline << "\n\n";

  int result =
      system(pipeline.c_str());

  if (result != 0) {

    cerr << "Camera pipeline failed\n";

    return 1;
  }

  return 0;
}