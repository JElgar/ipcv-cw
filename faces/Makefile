default: build 

build: face.cpp
	g++ $? -o $@ -Wl,-rpath,/usr/local/lib `pkg-config --cflags --libs opencv4`
