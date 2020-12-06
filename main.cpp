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


const float THRESHOLD = 0.6;

/** Function Headers */
std::vector<Rect> detectFaces( Mat frame );

/** Global variables */
String cascade_name = "frontalface.xml";
//String cascade_name = "dartcascade/cascade.xml";
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


Mat convolution(cv::Mat &image, cv::Mat &kernel) {

    cv::Mat output;
    image.copyTo(output);

	int kernelRadiusX = (kernel.size[0] - 1 ) / 2;
	int kernelRadiusY = (kernel.size[1] - 1 ) / 2;
	
    cv::Mat paddedInput;
	cv::copyMakeBorder( image, paddedInput, 
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE );
    
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
			double sum = 0.0;
			for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ )
			{
				for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ )
				{
					// find the correct indices we are using
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// get the values from the padded image and the kernel
					int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
					double kernalval = kernel.at<double>( kernelx, kernely );

					// do the multiplication
					sum += imageval * kernalval;							
				}
			}
			// set the output value as the sum of the convolution
			output.at<uchar>(i, j) = (uchar) sum;
        }
    }
    
    return output;
}

void sobel(cv::Mat &image) {
    
    double kernelXData[]={-1, 0, 1, -2, 0, 2, -1, 0, 1};
    double kernelYData[]={-1, -2, -1, 0, 0, 0, 1, 2, 1};
    
    cv::Mat dx;
    image.copyTo(dx);
    cv::Mat kernelX(3,3,CV_16S, kernelXData);
    convolution(dx, kernelX);
    
    cv::Mat dy;
    image.copyTo(dy);
    cv::Mat kernelY(3,3,CV_16S, kernelYData);
    convolution(image, kernelY);
    
    cv::Mat dx2;
    cv::Mat dy2;
    cv::pow(dx, 2.0, dx2);
    cv::pow(dy, 2.0, dy2);

    cv::Mat gradient2 = dx2 + dy2;

    cv::Mat gradient;
    cv::Mat imagec;

    gradient2.convertTo(imagec, CV_32F);
    cv::pow(imagec, 0.5, gradient);
    
    cv::imwrite("grad.png", gradient);
    
    cv::Mat dyDividedByDx;
    cv::divide(dy, dx, dyDividedByDx);
    cv::Mat direction;
    image.copyTo(direction);

    for (int y = 0; y < image.rows; ++y) {
      for (int x = 0; x < image.cols; ++x) {
        direction.at<uchar>(y,x) = std::atan(dyDividedByDx.at<uchar>(y, x)) - M_PI/2;
      }
    }
    
    cv::Mat directionOutput;
    cv::multiply(direction, 10, directionOutput);
    cv::imwrite("grad-direction.png", direction);
}

void hough(Mat &input) {

  Mat output, input_gray, input_edges;
  input.copyTo(output);
  cvtColor( input, input_gray, CV_BGR2GRAY);

  Canny(input_gray, input_edges, 50, 200, 3);

  imwrite("edges.jpg", input_edges);

  vector<Vec2f> lines;
  HoughLines(input_edges, lines, 1, CV_PI/180, 150, 0, 0 );

  // Draw the lines
  for( size_t i = 0; i < lines.size(); i++ )
  {
      float rho = lines[i][0], theta = lines[i][1];
      Point pt1, pt2;
      double a = cos(theta), b = sin(theta);
      double x0 = a*rho, y0 = b*rho;
      pt1.x = cvRound(x0 + 1000*(-b));
      pt1.y = cvRound(y0 + 1000*(a));
      pt2.x = cvRound(x0 - 1000*(-b));
      pt2.y = cvRound(y0 - 1000*(a));
      line( output, pt1, pt2, Scalar(0,0,255), 2, 8);
  }
  cv::imwrite("hough.jpg", output);
}


void houghCircles(Mat &input) {

  Mat output, input_gray;
  input.copyTo(output);
  cvtColor( input, input_gray, CV_BGR2GRAY);

  medianBlur(input_gray, input_gray, 5);

  vector<Vec3f> circles;
  HoughCircles(input_gray, circles, CV_HOUGH_GRADIENT, 20, input.rows/16, 100, 30);

  // Draw the lines
  for( size_t i = 0; i < circles.size(); i++ )
  {
    Vec3i c = circles[i];
    Point center = Point(c[0], c[1]);
    // circle center
    circle( input, center, 1, Scalar(0,100,100), 3, 8);
    // circle outline
    int radius = c[2];
    circle( input, center, radius, Scalar(255,0,255), 3, 8);
  }
  cv::imwrite("houghCircles.jpg", input);
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

  while(getline(infile, line)) {
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
  return (detected_rect & true_rect).area() / (float)(detected_rect | true_rect).area();
}


int number_of_correctly_detected_faces(vector<Rect> detected_rects, vector<Rect> true_rects) {
  int number_of_detected_faces = 0; 
  for (int i = 0; i < true_rects.size(); i++) {
    float max_iou = 0;
    for (int j = 0; j < detected_rects.size(); j++) {
      float iou = intersection_over_union(detected_rects[j], true_rects[i]);
      //cout << "IOU: "<< iou << endl;
      if (iou > max_iou) {
        max_iou = iou;
      }
    }
    if (max_iou > THRESHOLD) {
      number_of_detected_faces++;
    } else {
      cout << "Image at index " << i << " rejected" << endl;
    }
  }
  return number_of_detected_faces;
}


float true_positive_rate(vector<Rect> detected_rects, vector<Rect> true_rects) {
  if (true_rects.size() == 0) {
    cout << "No true faces provided" << endl;
    return 1;
  }
  return (float)number_of_correctly_detected_faces(detected_rects, true_rects) / (float)true_rects.size();
}


float f1_score(vector<Rect> detected_rects, vector<Rect> true_rects) {
  float recall = true_positive_rate(detected_rects, true_rects);
  float precision = number_of_correctly_detected_faces(detected_rects, true_rects) / (float)detected_rects.size();
  if (precision + recall == 0) {
    return 0;
  }
  return (float)2 * (precision * recall) / (precision + recall);
}


/** @function draw */
void draw(Rect rect, Mat frame) {
    rectangle(frame, rect, Scalar( 0, 0, 255 ), 2);
}


void draw(vector<Rect> rects, Mat frame) {
  for ( int i = 0; i < rects.size(); i++ ) {
    draw(rects[i], frame);
  }
}


/** @function main */
int main( int argc, const char** argv )
{
    // 0. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

    // 0.2 do the things
    //hough(frame);
    //houghCircles(frame);

    if (frame.empty()) {
      printf("Error loading image\n");
      return -1;
    }
    
    // 1. Setup true faces
    vector<Rect> true_faces;
    if (argc > 2) {
      true_faces = get_true_face(argv[2]);
    }


	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

    // 3. Detect Faces and Display Result
    vector<Rect> detected_faces = detectFaces( frame );

    // Sobel
    Mat sobel_output;
    frame.copyTo(sobel_output);
    cv::Sobel(frame, sobel_output, CV_16S, 1, 1);
    sobel(frame);
	imwrite( "sobel.jpg", sobel_output );

    // 4 Draw true faces
    draw(true_faces, frame);
    
    // 5. Draw box around faces found
    draw(detected_faces, frame);
  
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


