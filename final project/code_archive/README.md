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

## Compile main.cpp

g++ -std=c++11 main.cpp -o main \
  $(pkg-config --cflags opencv4) \
  $(pkg-config --libs opencv4) \
  -lOpenCL \
  -Wl,--allow-shlib-undefined