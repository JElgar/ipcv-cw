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
      if (gradMag.at<float>(y, x) >= 255) {
        
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
  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      for (int r = minRadius; r < maxRadius; r++) {
        if (houghSpace[x][y][r] > threshold) {
          circles.push_back(cv::Vec3i(x, y, r));
        }
      }
    }
  }
  
  std::cout << "Drew circles" << std::endl;

  return circles;
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

    std::cout << "REady" << std::endl;
    std::vector<cv::Vec3i> circles = houghCircles(image_gray, 18);
    std::cout << "Circles length:" << circles.size() << std::endl;
    
    cv::Mat output;
    image.copyTo(output);
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
  
    std::cout << "Alls good" << std::endl;

    // free memory
    image.release();

    return 0;
}

