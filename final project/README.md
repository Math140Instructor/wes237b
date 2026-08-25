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