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


std::vector<std::string> split(std::string line, std::string delimiter) {
  size_t pos = 0;
  std::vector<std::string> tokens;
  while ((pos = line.find(delimiter)) != std::string::npos) {
      tokens.push_back(line.substr(0, pos));
      line.erase(0, pos + delimiter.length());
  }
  tokens.push_back(line);
  return tokens;
}


cv::Mat convolution(cv::Mat &input, cv::Mat &kernel) {

    cv::Mat output;
    output.create(input.size(), CV_32FC1);

	int kernelRadiusX = (kernel.size[0] - 1 ) / 2;
	int kernelRadiusY = (kernel.size[1] - 1 ) / 2;
	
    cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput, 
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE );
    
    cv::imwrite("borderimage.png", paddedInput);

    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
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
			output.at<float>(i, j) = (float) sum;
        }
    }

    return output;
}

cv::Mat imageDx(cv::Mat &input) {
  cv::Mat kernelX = (cv::Mat_<float>(3,3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);

  cv::Mat dx = convolution(input, kernelX);
  cv::imwrite("dx.png", dx);
  return dx;
}

cv::Mat imageDy(cv::Mat &input) {
  cv::Mat kernelY = (cv::Mat_<float>(3,3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
  cv::Mat dy = convolution(input, kernelY);
  std::cout << "Writing dy" << std::endl;
  cv::imwrite("dy.png", dy);
  return dy;
}


cv::Mat gradientMagnitude (cv::Mat &input) {
  std::cout << "In grad mag" << std::endl;
    cv::Mat grad;
    grad.create(input.size(), CV_32FC1);

    cv::Mat dx = imageDy(input);
    cv::Mat dy = imageDy(input);

	for (int x = 0; x < input.cols; x++) {
	  for (int y = 0; y < input.rows; y++) {
		  grad.at<float>(y,x) = sqrt( pow(dx.at<float>(y,x), 2) + pow(dy.at<float>(y,x), 2) );	
	  }
	}
    return grad;
}

cv::Mat gradientDirection (cv::Mat &input) {
    
    cv::Mat grad;
    grad.create(input.size(), CV_32FC1);

    cv::Mat dx = imageDx(input);
    cv::Mat dy = imageDy(input);

	for (int x = 0; x < input.cols; x++) {
	  for (int y = 0; y < input.rows; y++) {
		  grad.at<float>(y,x) = std::atan( dy.at<float>(y,x) / (float) dx.at<float>(y,x) );	
	  }
	}
    return grad;
}

void scaling( int ***hough, int width, int height, int maxRadius){
	int max = 0;
	// find the max
	for (int x=0; x< width; x++){
		for(int y=0; y< height; y++){
			for (int r=0; r< maxRadius; r++){
				if (hough[x][y][r] > max){
					max = hough[x][y][r];
				}	
			}
		}
	}
	// scale the thing
	for (int x=0; x< width; x++){
		for(int y=0; y< height; y++){
			for (int r=0; r< maxRadius; r++){
				hough[x][y][r] = ( hough[x][y][r] * 255 ) / max ;	
			}
		}
	}
}


std::vector<cv::Vec3i> houghCircles (cv::Mat &input, int threshold = 20) {
  
  cv::Mat input_edges, input_gray, magnitude;
  input_gray = input;
  cv::Mat gradMag = gradientMagnitude(input_gray);
  cv::Mat gradDir = gradientDirection(input_gray);

  int height = input.rows, width = input.cols;

  int maxRadius = 120;
  int minRadius = 30;
  int rangeRadius = maxRadius - minRadius;

  int thetaErrorRange = 13;

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
      if (gradMag.at<float>(y, x) >= 80) {
        
        // For every radius we are checking
        for (int r = minRadius; r < maxRadius; r++) {
          for (int t = gradDir.at<float>(y,x) - thetaErrorRange; t < gradDir.at<float>(y,x) + thetaErrorRange; t++) {

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
  //scaling(houghSpace, width, height, maxRadius);
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
  while (true) {

    int currentMax = 0;
    cv::Vec3i circleAtMax;

    for (int x = 0; x < width; x++) {
      for (int y = 0; y < height; y++) {
        for (int r = minRadius; r < maxRadius; r++) {
          if (houghSpace[x][y][r] > currentMax) {
            currentMax = houghSpace[x][y][r];
            circleAtMax = cv::Vec3i(x, y, r);
          }
        }
      }
    }

    // Black out pixles around the peak to avoid clustering
    if (currentMax > threshold) {
      circles.push_back(circleAtMax);
      int blackOutRadius = 20;
      for ( float x = circleAtMax[0] - blackOutRadius; x <= circleAtMax[0] + blackOutRadius; x++) {
        for ( float y = circleAtMax[1] - blackOutRadius; y <= circleAtMax[1] + blackOutRadius; y++) {
          for ( float r = circleAtMax[2] - blackOutRadius; r <= circleAtMax[2] + blackOutRadius; r++) {
            if (x >= 0 && y>=0 && r >= 0 && x < width && y < height && r < maxRadius) {
              houghSpace[(int)x][(int)y][(int)r] = 0.0f;
            }
          }
        }
      }
    }
    else {
      break;
    }

  }
  
  std::cout << "Drew circles" << std::endl;

  return circles;
}

std::vector<cv::Rect> get_true_face(std::string path) {

  // Read in file
  std::ifstream infile(path.c_str());
 
  if (!infile) {
    std::cerr << "Can't open input file " << path << std::endl; 
  }

  std::string line;
  std::string token;
  std::vector<cv::Rect> faces;

  while(getline(infile, line)) {
    std::vector<std::string> tokens = split(line, ",");
    
    int x = std::stoi(tokens[0]);
    int y = std::stoi(tokens[1]);
    int width = std::stoi(tokens[2]);
    int height = std::stoi(tokens[3]);
    
    faces.push_back(cv::Rect(x, y, width, height));
  }

  return faces;
}

void draw(cv::Rect rect, cv::Mat frame, cv::Scalar colour) {
    rectangle(frame, rect, colour, 2);
}

void draw(std::vector<cv::Rect> rects, cv::Mat frame, cv::Scalar colour) {
  for ( int i = 0; i < rects.size(); i++ ) {
    draw(rects[i], frame, colour);
  }
}

cv::Rect circleToRect(cv::Vec3i circle) {
  return cv::Rect(cv::Point(circle[0] - circle[2], circle[1] - circle[2]), cv::Point(circle[0] + circle[2], circle[1] + circle[2]));
}

float intersection_over_union(cv::Rect detected_rect, cv::Rect true_rect) {
  return (detected_rect & true_rect).area() / (float)(detected_rect | true_rect).area();
}

float intersection_over_union(cv::Vec3i circle, cv::Rect rect) {
  cv::Rect circle_rect = circleToRect(circle);
  return std::max(intersection_over_union(circle_rect, rect), intersection_over_union(rect, circle_rect));
}


int number_of_correctly_detected_faces(std::vector<cv::Rect> detected_rects, std::vector<cv::Rect> true_rects, float threshold = 0.6) {
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
    if (max_iou > threshold) {
      number_of_detected_faces++;
    } else {
      std::cout << "Image at index " << i << " rejected" << std::endl;
    }
  }
  return number_of_detected_faces;
}


float true_positive_rate(std::vector<cv::Rect> detected_rects, std::vector<cv::Rect> true_rects) {
  if (true_rects.size() == 0) {
    std::cout << "No true faces provided" << std::endl;
    return 1;
  }
  return (float)number_of_correctly_detected_faces(detected_rects, true_rects) / (float)true_rects.size();
}


float f1_score(std::vector<cv::Rect> detected_rects, std::vector<cv::Rect> true_rects) {
  float recall = true_positive_rate(detected_rects, true_rects);
  float precision = number_of_correctly_detected_faces(detected_rects, true_rects) / (float)detected_rects.size();
  if (precision + recall == 0) {
    return 0;
  }
  return (float)2 * (precision * recall) / (precision + recall);
}

std::vector<cv::Rect> voilaJonesDartDetection(cv::Mat &input) {
  cv::String cascade_name = "dartcascade/cascade.xml";
  cv::CascadeClassifier cascade;
  if( !cascade.load( cascade_name ) ){ 
    printf("--(!)Error loading\n"); 
    std::vector<cv::Rect> rects;
    return rects;
  }
    
  std::vector<cv::Rect> faces;
  cascade.detectMultiScale( input, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, cv::Size(50, 50), cv::Size(500,500) );
  return faces;
}

std::vector<cv::Rect> detectDartboards(cv::Mat image_gray, float threshold = 0.5) {

  // Do vj and hough and store results
  std::vector<cv::Vec3i> hough_boards = houghCircles(image_gray, 35);
  std::vector<cv::Rect> vj_boards = voilaJonesDartDetection(image_gray);

  // Find IOU of hough_boards and vj_boards and  
  std::vector<cv::Rect> combined_boards; 
  for (cv::Rect vj_board : vj_boards) {
    float max_iou = 0;
    for (cv::Vec3i h_board : hough_boards) {
      float iou = intersection_over_union(h_board, vj_board);
      //cout << "IOU: "<< iou << endl;
      if (iou > max_iou) {
        max_iou = iou;
      }
    }
    std::cout << max_iou << std::endl;
    if (max_iou > threshold) {
      combined_boards.push_back(vj_board);
    } 
  }
  return combined_boards;
}


int
main (int argc, char **argv)
{

    cv::Mat image, image_gray;
    image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);

    cv::cvtColor( image, image_gray, CV_BGR2GRAY);
    cv::equalizeHist( image_gray, image_gray);
    //cv::medianBlur(image_gray, image_gray, 5);

    cv::Mat gradD = gradientDirection(image_gray);
    cv::Mat gradM = gradientMagnitude(image_gray);

    cv::imwrite("graddirection.png", gradD);
    cv::imwrite("gradmag.png", gradM);

    // -- Hough -- //
    std::vector<cv::Vec3i> circles = houghCircles(image_gray, 35);
    cv::Mat circlesOutput;
    image.copyTo(circlesOutput);
    for (cv::Vec3i circle : circles) {
      std::cout << circle[2] << std::endl;
      cv::Rect circle_rect = circleToRect(circle);
      cv::Point center = cv::Point(circle[0], circle[1]);
      // circle center
      cv::circle( circlesOutput, center, 1, cv::Scalar(0,100,100), 3, 8);
      // circle outline
      int radius = circle[2];
      cv::circle( circlesOutput, center, radius, cv::Scalar(255,0,255), 3, 8);
      cv::rectangle(circlesOutput, circle_rect , cv::Scalar(255, 0 , 255), 2);
    }
    std::cout << "Circles length:" << circles.size() << std::endl;
    cv::imwrite("cirlce-hough-output.jpg", circlesOutput);
   
    // -- Viola Jones -- //
    std::vector<cv::Rect> boards = voilaJonesDartDetection(image_gray);
    cv::Mat vjOutput;
    image.copyTo(vjOutput);
    draw(boards, vjOutput, cv::Scalar(0, 0, 255));
    std::cout << "Boards length:" << boards.size() << std::endl;
    cv::imwrite("vj-output.jpg", vjOutput);

    std::vector<cv::Rect> finalBoards = detectDartboards(image_gray);
    cv::Mat finalOutput;
    image.copyTo(finalOutput);
    draw(finalBoards, finalOutput, cv::Scalar(0, 255, 0));
    cv::imwrite("final-output.jpg", finalOutput);

    // free memory
    image.release();

    return 0;
}

