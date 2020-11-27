/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
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

using namespace std;
using namespace cv;


const float THRESHOLD = 0.7;

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

  // Read in file
  ifstream infile(path.c_str());
 
  if (!infile) {
    cerr << "Can't open input file " << path << endl; 
  }

  string line;
  string token;
  vector<Rect> faces;

  cout << "Getting tokens " << endl;
  while(getline(infile, line)) {
    cout << "Wokring on line " << line << endl;
    vector<string> tokens = split(line, ",");
    
    int x = sti(tokens[0]);
    int y = sti(tokens[1]);
    int width = sti(tokens[2]);
    int height = sti(tokens[3]);
    
    faces.push_back(Rect(x, y, width, height));
  }

  return faces;
}


float intersection_over_union(Rect detected_rect, Rect true_rect) {

  Point topRightIntersectionPoint = Point(max(detected_rect.x, true_rect.x), max(detected_rect.y, true_rect.y));
  Point bottomRightIntersectionPoint = Point(min(detected_rect.x + detected_rect.width, true_rect.x + true_rect.width), min(detected_rect.y + detected_rect.height, true_rect.y + true_rect.height));

  Rect intersection_rect = Rect(topRightIntersectionPoint, bottomRightIntersectionPoint);

  return intersection_rect.area() / (float)(detected_rect.area() + true_rect.area() - intersection_rect.area());
}


int number_of_correctly_detected_faces(vector<Rect> detected_rects, vector<Rect> true_rects) {
  int number_of_detected_faces = 0; 
  for (int i = 0; i < true_rects.size(); i++) {
    float max_iou = 0;
    for (int j = 0; j < detected_rects.size(); j++) {
      float iou = intersection_over_union(detected_rects[j], true_rects[i]);
      if (iou > max_iou) {
        max_iou = iou;
      }
    }
    if (max_iou > THRESHOLD) {
      number_of_detected_faces++;
    }
  }
  std::cout << number_of_detected_faces << ", "<< true_rects.size() << endl;
  return number_of_detected_faces;
}


float true_positive_rate(vector<Rect> detected_rects, vector<Rect> true_rects) {
  return (float)number_of_correctly_detected_faces(detected_rects, true_rects) / (float)true_rects.size();
}


float f1_score(vector<Rect> detected_rects, vector<Rect> true_rects) {
  float recall = true_positive_rate(detected_rects, true_rects);
  float precision = number_of_correctly_detected_faces(detected_rects, true_rects) / (float)detected_rects.size();

  std::cout << recall << ", "<< precision << endl;
  return (float)2 * (precision * recall) / (precision + recall);
}


/** @function draw */
void draw(Rect rect, Mat frame) {
    rectangle(frame, rect, Scalar( 0, 0, 255 ), 2);
}


/** @function main */
int main( int argc, const char** argv )
{
    // 0. Setup true faces
    vector<Rect> true_faces = get_true_face(argv[2]);

    // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

    // 3. Detect Faces and Display Result
    vector<Rect> detected_faces = detectFaces( frame );

    // 4 Draw true faces
	for( int i = 0; i < true_faces.size(); i++ )
	{
      draw(true_faces[i], frame);
    }
    
    // 5. Draw box around faces found
	for( int i = 0; i < detected_faces.size(); i++ )
	{
		rectangle(frame, detected_faces[i], Scalar( 0, 255, 0 ), 2);
	}

    cout << "IOA: " << intersection_over_union(detected_faces[0], true_faces[0]) << endl;
    cout << "TPR: " << true_positive_rate(detected_faces, true_faces) << endl;
    cout << "F1: " << f1_score(detected_faces, true_faces) << endl;

	// 6. Save Result Image
	imwrite( "detected.jpg", frame );

	return 0;
}

/** @function detectAndDisplay */
vector<Rect> detectFaces( Mat frame )
{
	vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY);
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

    // 3. Print number of Faces found
	cout << faces.size() << endl;

    return faces;

}


