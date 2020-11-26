default: build 

build: faces/face.cpp
	g++ faces/face.cpp -o faces.o /usr/lib64/libopencv_core.so /usr/lib64/libopencv_highgui.so.2.4 /usr/lib64/libopencv_imgproc.so.2.4 /usr/lib64/libopencv_objdetect.so.2.4
