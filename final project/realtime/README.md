g++ -std=c++11 main.cpp -o main \
    $(pkg-config --cflags opencv4) \
    $(pkg-config --libs opencv4) \
    -Wl,--allow-shlib-undefined