THIS WORKS!!
gst-launch-1.0 -e qtiqmmfsrc camera=0 ! \
video/x-raw,format=NV12,width=1280,height=720,framerate=30/1 ! \
videoconvert ! \
x264enc tune=zerolatency ! \
video/x-h264,profile=baseline ! \
h264parse ! \
mp4mux ! \
filesink location=test.mp4

apt update
apt install -y gstreamer1.0-plugins-ugly gstreamer1.0-plugins-bad

## Build
g++ main.cpp -o glasses_test $(pkg-config --cflags --libs opencv4) -Wl,--allow-shlib-undefined

## Verify JSON container build successful
gst-inspect-1.0 qtiqmmfsrc

gst-launch-1.0 qtiqmmfsrc camera=0 ! \
video/x-raw,format=NV12,width=1280,height=720,framerate=30/1 ! \
fakesink

pkg-config --modversion opencv4


## Copy the video output to host
SSH IP and location of file to host current user
`scp rb3:~/test.mp4 .`


## Conversion CPU to GPU
```for (int gy = 0; gy < rotated.rows; gy++) {
    for (int gx = 0; gx < rotated.cols; gx++) {
        ...
    }
}```


Camera BGR
   |
   v
 UMat
   |
   +---- cvtColor() ----------> OpenCL
   |
CPU Haar face/eye detection
   |
   +---- resize() ------------> OpenCL
   |
   +---- warpAffine() --------> OpenCL
   |
   +---- multiply/add --------> OpenCL alpha blend
   |
   v
 output.mp4

 UMat is OpenCV's transparent API mechanism for keeping data in device-backed memory and allowing supported operations to select OpenCL implementations. The actual OpenCL device can be inspected through cv::ocl::Device, which is why I added the startup output.

 `g++ -std=c++11 main_cl.cpp -o glasses \
    $(pkg-config --cflags opencv4) \
    $(pkg-config --libs opencv4) \
     -Wl,--allow-shlib-undefined`

## main_gpu.cpp
`g++ -std=c++11 main_gpu.cpp -o glasses_gpu \
    $(pkg-config --cflags opencv4) \
    $(pkg-config --libs opencv4) \
    -Wl,--no-as-needed \
    /lib/aarch64-linux-gnu/libOpenCL.so.1 \
    -Wl,--allow-shlib-undefined`

OUTPUT
root@ubuntu:/workspaces/wes237b/final project# ./glasses_gpu 
./glasses_gpu: /lib/aarch64-linux-gnu/libOpenCL.so.1: no version information available (required by /lib/aarch64-linux-gnu/libavutil.so.58)
./glasses_gpu: /lib/aarch64-linux-gnu/libOpenCL.so.1: no version information available (required by /lib/aarch64-linux-gnu/libavutil.so.58)
OpenCL GPU device: QUALCOMM Adreno(TM) 643
OpenCL vendor: QUALCOMM
OpenCL version: OpenCL 3.0 Adreno(TM) 643
Compute units: 2
Max work-group size: 1024
OpenCL kernels compiled successfully.
failed to get driver name for fd 5
MESA-LOADER: failed to retrieve device information
failed to get driver name for fd 5
[ WARN:0@0.504] global ./modules/videoio/src/cap_gstreamer.cpp (1374) open OpenCV | GStreamer warning: unable to query duration of stream
[ WARN:0@0.504] global ./modules/videoio/src/cap_gstreamer.cpp (1405) open OpenCV | GStreamer warning: Cannot query video position: status=0, value=-1, duration=-1
RB3 camera opened.
Camera resolution: 1280x720
Recording processed video to output.mp4
Processed 30 frames | GPU gray launches: 30 | GPU alpha launches: 0 | frames with face: 8 | frames with 2 eyes: 0
Processed 60 frames | GPU gray launches: 60 | GPU alpha launches: 21 | frames with face: 34 | frames with 2 eyes: 21
Processed 90 frames | GPU gray launches: 90 | GPU alpha launches: 40 | frames with face: 56 | frames with 2 eyes: 40
Processed 120 frames | GPU gray launches: 120 | GPU alpha launches: 47 | frames with face: 71 | frames with 2 eyes: 47
Processed 150 frames | GPU gray launches: 150 | GPU alpha launches: 51 | frames with face: 76 | frames with 2 eyes: 51
Processed 180 frames | GPU gray launches: 180 | GPU alpha launches: 54 | frames with face: 81 | frames with 2 eyes: 54
Processed 210 frames | GPU gray launches: 210 | GPU alpha launches: 84 | frames with face: 111 | frames with 2 eyes: 84
Processed 240 frames | GPU gray launches: 240 | GPU alpha launches: 94 | frames with face: 124 | frames with 2 eyes: 94
Processed 270 frames | GPU gray launches: 270 | GPU alpha launches: 121 | frames with face: 151 | frames with 2 eyes: 121
Processed 300 frames | GPU gray launches: 300 | GPU alpha launches: 121 | frames with face: 152 | frames with 2 eyes: 121
root@ubuntu:/workspaces/wes237b/final project# 