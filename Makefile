default: build 

build: main2.cpp
	g++ main2.cpp --std=c++11 -o main.o /usr/lib64/libopencv_core.so /usr/lib64/libopencv_highgui.so.2.4 /usr/lib64/libopencv_imgproc.so.2.4 /usr/lib64/libopencv_objdetect.so.2.4

debug: main2.cpp
	g++ main2.cpp -g --std=c++11 -o main.o /usr/lib64/libopencv_core.so /usr/lib64/libopencv_highgui.so.2.4 /usr/lib64/libopencv_imgproc.so.2.4 /usr/lib64/libopencv_objdetect.so.2.4
