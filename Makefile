default: build 

build: src/main.cpp
	g++ src/main.cpp src/hough.cpp src/utils.cpp --std=c++11 -o main.o /usr/lib64/libopencv_core.so /usr/lib64/libopencv_highgui.so.2.4 /usr/lib64/libopencv_imgproc.so.2.4 /usr/lib64/libopencv_objdetect.so.2.4

debug: src/main.cpp
	g++ src/main.cpp src/hough.cpp src/utils.cpp -g --std=c++11 -o main.o /usr/lib64/libopencv_core.so /usr/lib64/libopencv_highgui.so.2.4 /usr/lib64/libopencv_imgproc.so.2.4 /usr/lib64/libopencv_objdetect.so.2.4
