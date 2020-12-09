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

#include <sstream>
#include <fstream>
#include <string>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <algorithm>

using namespace std;

const float THRESHOLD = 0.6;

/** Function Headers */
std::vector<cv::Rect> detectFaces( cv::Mat frame );

/** Global variables */
//String cascade_name = "frontalface.xml";
cv::String cascade_name = "dartcascade/cascade.xml";
cv::CascadeClassifier cascade;

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

cv::Mat convolution(cv::Mat &image, cv::Mat &kernel) {
 
    cv::Mat output;
    output.create(image.size(), CV_64FC1);

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
					sum += (double)imageval * kernalval;							
				}
			}
			// set the output value as the sum of the convolution
			output.at<double>(i, j) = (double) sum;
        }
    }
    
    return output;
}


cv::Mat imageDx(cv::Mat &input) {
  double kernelXData[]={-1, 0, 1, -2, 0, 2, -1, 0, 1};

  cv::Mat kernelX(3,3, CV_16S, kernelXData);
  cv::Mat dx = convolution(input, kernelX);
  cv::imwrite("dx.png", dx);
  return dx;
}

cv::Mat imageDy(cv::Mat &input) {
  double kernelYData[]={-1, -2, -1, 0, 0, 0, 1, 2, 1};

  cv::Mat kernelY(3,3,CV_16S, kernelYData);
  cv::Mat dy = convolution(input, kernelY);
  cv::imwrite("dy.png", dy);
  return dy;
}

cv::Mat gradientMagnitude (cv::Mat &input) {
    cv::Mat grad;
    grad.create(input.size(), CV_64FC1);

    cv::Mat dx = imageDy(input);
    cv::Mat dy = imageDy(input);

	for (int x = 0; x < input.cols; x++) {
	  for (int y = 0; y < input.rows; y++) {
		  grad.at<uchar>(y,x) = sqrt( pow(dx.at<uchar>(y,x), 2) + pow(dy.at<uchar>(y,x), 2) );	
	  }
	}
    return grad;
}

cv::Mat gradientDirection (cv::Mat &input) {
    cv::Mat grad;
    grad.create(input.size(), CV_64FC1);

    cv::Mat dx = imageDx(input);
    cv::Mat dy = imageDy(input);

	for (int x = 0; x < input.cols; x++) {
	  for (int y = 0; y < input.rows; y++) {
		  grad.at<uchar>(y,x) = std::atan( dy.at<uchar>(y,x) / dx.at<uchar>(y,x) );	
	  }
	}
    return grad;
}

void sobel(cv::Mat &image) {
   
    cv::Mat dx = imageDx(image);
    cv::Mat dy = imageDy(image);
    
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

void nicolesScaling( int *** hough, int maxR, int cols, int rows ){
	cout<< " -> ... hough in progress " <<endl;
	int max = 0;
	// find the max
	for (int x=0; x< cols; x++){
		for(int y=0; y< rows; y++){
			for (int r=0; r< maxR; r++){
				if (hough[x][y][r] > max){
					max = hough[x][y][r];
				}	
			}
		}
	}
	// scale the thing
	for (int x=0; x< cols; x++){
		for(int y=0; y< rows; y++){
			for (int r=0; r< maxR; r++){
				hough[x][y][r] = ( hough[x][y][r] * 255 ) / max ;	
			}
		}
	}
}

vector<cv::Vec2f> houghLines(cv::Mat &input, int threshold = 500) {
  
  int numberOfAngles = 1800;

  double width = input.size().width, height = input.size().height;
  double imageHypot = std::hypot(width, height);

  double rhoMin = -imageHypot;
  double rhoMax = imageHypot + 1;
  double numberOfRadii = 2*imageHypot + 1;

  double thetaMax = CV_PI;
  double thetaStep = thetaMax / (double)numberOfAngles;

  // Setup
  cv::Mat input_edges, input_gray;
  cvtColor( input, input_gray, CV_BGR2GRAY);
  Canny(input_gray, input_edges, 50, 200, 3);
  cv::imwrite("input_edges_myhough.png", input_edges);

  cv::Mat houghSpace = cv::Mat::zeros(numberOfRadii, numberOfAngles, CV_32SC1);
  std::cout << "Hough space height: " << houghSpace.size().height << std::endl;
  std::cout << "Hough space width: " << houghSpace.size().width << std::endl;

  // For every pixel in the image 
  for (int y = 0; y < input.rows; y++) {
    for (int x = 0; x < input.cols; x++) {
      
      // If the pixel is not empty (i.e on an edge)
      if (input_edges.at<int>(y, x) != 0) {
        
        // For every angle
        for (int thetaIndex = 0; thetaIndex < numberOfAngles; thetaIndex++) {
          double theta = (double)thetaIndex * thetaStep;
          double rho = x * std::cos(theta) + y * std::sin(theta);
          rho += imageHypot;
          houghSpace.at<int>(rho, thetaIndex)++;
        }
      }
    }
  }

  cv::imwrite("myhough.png", houghSpace*5);
  vector<cv::Vec2f> houghLines;
  // Search through the accumulator and find the maximums
  for (int rho = 1; rho < numberOfRadii-1; rho++) {
    for (int theta = 1; theta < numberOfAngles-1; theta++) {
      int valueAtPoint = houghSpace.at<int>(rho,theta);
      if (    houghSpace.at<int>(rho,theta) > threshold 
          &&  houghSpace.at<int>(rho,theta) > houghSpace.at<int>(rho,theta-1)
          &&  houghSpace.at<int>(rho,theta) >= houghSpace.at<int>(rho,theta+1)
          &&  houghSpace.at<int>(rho,theta) > houghSpace.at<int>(rho-1,theta)
          &&  houghSpace.at<int>(rho,theta) >= houghSpace.at<int>(rho+1,theta)
      ) {
        std::cout << "line with rho: " << rho - imageHypot << std::endl;
        std::cout << "line with theta: " << theta * thetaStep<< std::endl;
        cv::Vec2f line = cv::Vec2f(theta * thetaStep, rho-imageHypot);
        houghLines.push_back(line);
      }
    }
  }

  cv::Mat output;
  input.copyTo(output);
  for( size_t i = 0; i < houghLines.size(); i++ )
  {
      float rho = houghLines[i][1];
      std::cout << "Drawing line with rho: " << rho << std::endl;
      float theta = houghLines[i][0];
      std::cout << "Drawing line with theta: " << theta << std::endl;
      cv::Point pt1, pt2;
      double a = cos(theta), b = sin(theta);
      double x0 = a*rho, y0 = b*rho;
      pt1.x = cvRound(x0 + 1000*(-b));
      pt1.y = cvRound(y0 + 1000*(a));
      pt2.x = cvRound(x0 - 1000*(-b));
      pt2.y = cvRound(y0 - 1000*(a));
      line( output, pt1, pt2, cv::Scalar(0,0,255), 2, 8);
  }
  cv::imwrite("myhough-output.png", output);

  return houghLines;
}

std::vector<cv::Vec3i> houghCircles (cv::Mat &input, int threshold = 30) {
 
  cv::Mat input_gray = input;
  Canny(input_gray, input_edges, 50, 200, 3);
  cv::imwrite("input_edges_myhoughcircles.png", input_edges);

  cv::Mat gradMag = gradientMagnitude(input_gray);
  cv::Mat gradDir = gradientDirection(input_gray);

  int height = input.rows, width = input.cols;

  int maxRadius = 100;
  int minRadius = 10;
  int rangeRadius = maxRadius - minRadius;

  // 3D array to store hough values
  std::cout << "Creating array" << std::endl;
  int ***houghSpace;
  houghSpace = new int**[width];
  for (int x = 0; x < width; x++) {
    houghSpace[x] = new int*[height];
    for (int y = 0; y < height; y++) {
      houghSpace[x][y] = new int[maxRadius];
    }
  }
  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      for (int r = 0; r < maxRadius; r++) {
        houghSpace[x][y][r] = 0;
      }
    }
  }
  std::cout << "Created array" << std::endl;

  std::cout << "Starting Hough" << std::endl;
  // For every pixel in the image 
  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      
      // If the pixel is not empty (i.e on an edge)
      if (gradMag.at<double>(y, x) == 255) {
        
        // For every radius we are checking
        for (int r = minRadius; r < maxRadius; r++) {
          for (int t = gradDir.at<uchar>(y,x) - 10; t < gradDir.at<uchar>(y,x) + 10; t++) {

            int xVal = x - (int) (r * std::cos(t));
            int yVal = y - (int) (r * std::sin(t));
            if (xVal >= 0 && xVal < width && yVal >= 0 && yVal < height) {
              houghSpace[xVal][yVal][r]++;
            }

            xVal = x + (int) (r * std::cos(t));
            yVal = y + (int) (r * std::sin(t));
            if (xVal >= 0 && xVal < width && yVal >= 0 && yVal < height) {
              houghSpace[xVal][yVal][r]++;
            }
          }
        }
      }
    }
  }
  std::cout << "Got hough space" << std::endl;

  std::cout << "Scaling" << std::endl;
  //nicolesScaling(houghSpace, maxRadius, width, height);
  std::cout << "Scaled" << std::endl;

  // Covert the houghSpace into an image so we can have a look
  cv::Mat houghSpaceImage = cv::Mat::zeros(width, height, CV_32SC1);
  for (int x = 0; x < width; x++){
    for (int y = 0; y < height; y++){
	  for (int r = minRadius; r < maxRadius; r++){
	  	houghSpaceImage.at<int>(y,x) += houghSpace[x][y][r];
	  }
    }
  }
  std::cout << "Got a iamge" << std::endl;
  cv::imwrite("cirlce-hough-space.jpg", houghSpaceImage);

  std::vector<cv::Vec3i> circles;
  // For every pixel in the hough space
  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      for (int r = minRadius; r < maxRadius; r++) {
        if (houghSpace[x][y][r] > threshold) {
          circles.push_back(cv::Vec3i(x, y, r));
        }
      }
    }
  }

  cv::Mat output;
  input.copyTo(output);
  for (cv::Vec3i circle : circles) {
    cv::Point center = cv::Point(circle[0], circle[1]);
    // circle center
    cv::circle( output, center, 1, cv::Scalar(0,100,100), 3, 8);
    // circle outline
    int radius = circle[2];
    cv::circle( output, center, radius, cv::Scalar(255,0,255), 3, 8);
  }

  std::cout << "Circles length:" << circles.size() << std::endl;
  cv::imwrite("cirlce-hough-output.jpg", output);

  return circles;
}

void hough(cv::Mat &input) {

  cv::Mat output, input_gray, input_edges;
  input.copyTo(output);
  cvtColor( input, input_gray, CV_BGR2GRAY);

  Canny(input_gray, input_edges, 50, 200, 3);

  imwrite("edges.jpg", input_edges);

  vector<cv::Vec2f> lines;
  cv::HoughLines(input_edges, lines, 1, CV_PI/180, 150, 0, 0 );

  // Draw the lines
  for( size_t i = 0; i < lines.size(); i++ )
  {
      float rho = lines[i][0], theta = lines[i][1];
      cv::Point pt1, pt2;
      double a = cos(theta), b = sin(theta);
      double x0 = a*rho, y0 = b*rho;
      pt1.x = cvRound(x0 + 1000*(-b));
      pt1.y = cvRound(y0 + 1000*(a));
      pt2.x = cvRound(x0 - 1000*(-b));
      pt2.y = cvRound(y0 - 1000*(a));
      line( output, pt1, pt2, cv::Scalar(0,0,255), 2, 8);
  }
  cv::imwrite("hough.jpg", output);
}


void cvHoughCircles(cv::Mat &input) {

  cv::Mat output, input_gray;
  input.copyTo(output);
  cvtColor( input, input_gray, CV_BGR2GRAY);

  medianBlur(input_gray, input_gray, 5);

  vector<cv::Vec3f> circles;
  HoughCircles(input_gray, circles, CV_HOUGH_GRADIENT, 20, input.rows/16, 100, 30);

  // Draw the lines
  for( size_t i = 0; i < circles.size(); i++ )
  {
    cv::Vec3i c = circles[i];
    cv::Point center = cv::Point(c[0], c[1]);
    // circle center
    circle( input, center, 1, cv::Scalar(0,100,100), 3, 8);
    // circle outline
    int radius = c[2];
    circle( input, center, radius, cv::Scalar(255,0,255), 3, 8);
  }
  cv::imwrite("houghCircles.jpg", input);
}

/** path - Given a path to a file read in lines 1 by 1
 * Each line should consist of <x>,<y>,<widht>,<height>
 */
std::vector<cv::Rect> get_true_face(std::string path) {

  // Read in file
  ifstream infile(path.c_str());
 
  if (!infile) {
    cerr << "Can't open input file " << path << endl; 
  }

  string line;
  string token;
  vector<cv::Rect> faces;

  while(getline(infile, line)) {
    vector<string> tokens = split(line, ",");
    
    int x = sti(tokens[0]);
    int y = sti(tokens[1]);
    int width = sti(tokens[2]);
    int height = sti(tokens[3]);
    
    faces.push_back(cv::Rect(x, y, width, height));
  }

  return faces;
}


float intersection_over_union(cv::Rect detected_rect, cv::Rect true_rect) {
  return (detected_rect & true_rect).area() / (float)(detected_rect | true_rect).area();
}


int number_of_correctly_detected_faces(vector<cv::Rect> detected_rects, vector<cv::Rect> true_rects) {
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


float true_positive_rate(vector<cv::Rect> detected_rects, vector<cv::Rect> true_rects) {
  if (true_rects.size() == 0) {
    cout << "No true faces provided" << endl;
    return 1;
  }
  return (float)number_of_correctly_detected_faces(detected_rects, true_rects) / (float)true_rects.size();
}


float f1_score(vector<cv::Rect> detected_rects, vector<cv::Rect> true_rects) {
  float recall = true_positive_rate(detected_rects, true_rects);
  float precision = number_of_correctly_detected_faces(detected_rects, true_rects) / (float)detected_rects.size();
  if (precision + recall == 0) {
    return 0;
  }
  return (float)2 * (precision * recall) / (precision + recall);
}


void draw(cv::Rect rect, cv::Mat frame, cv::Scalar colour) {
    rectangle(frame, rect, colour, 2);
}

void draw(vector<cv::Rect> rects, cv::Mat frame, cv::Scalar colour) {
  for ( int i = 0; i < rects.size(); i++ ) {
    draw(rects[i], frame, colour);
  }
}

/** @function main */
int main( int argc, const char** argv )
{
    // 0. Read Input Image
    cv::Mat frame = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);

    // 0.2 do the things
    houghLines(frame);
    houghCircles(frame);
    hough(frame);
    //cvHoughCircles(frame);

    if (frame.empty()) {
      printf("Error loading image\n");
      return -1;
    }
    
    // 1. Setup true faces
    vector<cv::Rect> true_faces;
    if (argc > 2) {
      true_faces = get_true_face(argv[2]);
    }


	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

    // 3. Detect Faces and Display Result
    vector<cv::Rect> detected_faces = detectFaces( frame );

    // Sobel
    cv::Mat sobel_output;
    frame.copyTo(sobel_output);
    cv::Sobel(frame, sobel_output, CV_16S, 1, 1);
    sobel(frame);
	imwrite( "sobel.jpg", sobel_output );

    // 4 Draw true faces
    //draw(true_faces, frame, Scalar(0, 255, 0));
    
    // 5. Draw box around faces found
    //draw(detected_faces, frame, Scalar(0, 0, 255));
  
    cout << "TPR: " << true_positive_rate(detected_faces, true_faces) << endl;
    cout << "F1: " << f1_score(detected_faces, true_faces) << endl;

	//// 6. Save Result Image
	//imwrite( "detected.jpg", frame );

	return 0;
}

/** @function detectAndDisplay */
vector<cv::Rect> detectFaces( cv::Mat frame )
{
	vector<cv::Rect> faces;
    cv::Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY);
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, cv::Size(50, 50), cv::Size(500,500) );

    // 3. Print number of Faces found
	cout << faces.size() << endl;

    return faces;

}


