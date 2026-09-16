# Final Project
This code supports both real-time facial recognition and benchmarking of OpenCV CPU and OpenCL GPU video processing performance in C++ for the WES 267B final project.

## Real-time 

The following commands build and run the real-time facial feature recognition software. The program runs on the CPU by default, and pressing **G** toggles between CPU and GPU processing.

`cd realtime`

`g++ -std=c++11 main.cpp -o main \
  $(pkg-config --cflags opencv4) \
  $(pkg-config --libs opencv4) \
  -lOpenCL \
  -Wl,--allow-shlib-undefined`
  
`./main`

## Benchmarking

We created a `testbench` directory that captures a short baseline video and then performs post-processing on the same video using both the CPU and GPU, allowing for a consistent performance comparison.

`cd testbench`

`g++ -std=c++11 main.cpp -o main \
  $(pkg-config --cflags opencv4) \
  $(pkg-config --libs opencv4) \
  -lOpenCL \
  -Wl,--allow-shlib-undefined`

To change the default recording duration, provide the desired duration in seconds as a command-line argument:

`./main 5`

This example records for 5 seconds.
