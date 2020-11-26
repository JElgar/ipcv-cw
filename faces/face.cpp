/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

#include <sstream>
#include <fstream>
#include <string>
//#include <algorithm>

using namespace std;
using namespace cv;


/** Function Headers */
std::vector<Rect> detectFaces( Mat frame );

/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;

int sti(string val) {
  return atoi(val.c_str());
}

std::vector<string> split(string line, string delimiter) {
  size_t pos = 0;
  std::vector<string> tokens;
  while ((pos = line.find(delimiter)) != std::string::npos) {
      tokens.push_back(line.substr(0, pos));
      line.erase(0, pos + delimiter.length());
  }
  tokens.push_back(line);
  return tokens;
}

/** path - Given a path to a file read in lines 1 by 1
 * Each line should consist of <x>,<y>,<widht>,<height>
 */
std::vector<Rect> get_true_face(std::string path) {

  // Copy path to char array
  char path_char[path.length() + 1];
  strcpy(path_char, path.c_str());

  // Read in file
  ifstream infile(path_char);
 
  if (!infile) {
    cerr << "Can't open input file " << path_char << endl; 
  }

  std::string line;
  std::string token;
  std::vector<Rect> faces;

  std::cout << "Getting tokens " << endl;
  while(std::getline(infile, line)) {
    std::cout << "Wokring on line " << line << endl;
    std::vector<string> tokens = split(line, ",");
    
    int x = sti(tokens[0]);
    int y = sti(tokens[1]);
    int width = sti(tokens[2]);
    int height = sti(tokens[3]);
    
    faces.push_back(Rect(x, y, width, height));
  }

  return faces;
}


/** @function draw */
void draw(Rect rect, Mat frame) {
    rectangle(frame, Point(rect.x, rect.y), Point(rect.x + rect.width, rect.y + rect.height), Scalar( 0, 0, 255 ), 2);
}


/** @function main */
int main( int argc, const char** argv )
{
    // 0. Setup true faces
    std::vector<Rect> true_faces = get_true_face(argv[2]);
    true_faces.push_back(Rect(Point(327, 96), Point(492, 278)));

    // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

    // 3. Detect Faces and Display Result
    std::vector<Rect> detected_faces = detectFaces( frame );

    // 4 Draw true faces
	for( int i = 0; i < true_faces.size(); i++ )
	{
      draw(true_faces[i], frame);
    }
    
    // 5. Draw box around faces found
	for( int i = 0; i < detected_faces.size(); i++ )
	{
		rectangle(frame, Point(detected_faces[i].x, detected_faces[i].y), Point(detected_faces[i].x + detected_faces[i].width, detected_faces[i].y + detected_faces[i].height), Scalar( 0, 255, 0 ), 2);
	}


	// 4. Save Result Image
	imwrite( "detected.jpg", frame );

	return 0;
}

/** @function detectAndDisplay */
std::vector<Rect> detectFaces( Mat frame )
{
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY);
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

       // 3. Print number of Faces found
	std::cout << faces.size() << std::endl;

    return faces;

}


