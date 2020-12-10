#include "hough.h"

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


std::vector<cv::Vec3i> houghCircles (cv::Mat &input, int threshold, bool drawHoughSpace) {
  
  cv::Mat input_edges, input_gray, magnitude;
  input_gray = input;
  cv::Mat gradMag = gradientMagnitude(input_gray);
  cv::Mat gradDir = gradientDirection(input_gray);

  int height = input.rows, width = input.cols;

  int maxRadius = 120;
  int minRadius = 35;
  int rangeRadius = maxRadius - minRadius;

  int thetaErrorRange = 13;

  // Instead of using a thersholded magnitude image here we simply check the value is above a give threshold 
  int edgeThreshold = 80; 

  // -- Initalize a 3D array of size width height maxRadius to store the hough space -- //
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

  // -- Generate the hough space -- //
  // For every pixel in the image 
  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      
      // If the pixel is not empty (i.e on an edge)
      if (gradMag.at<float>(y, x) >= edgeThreshold) {
        
        // For every radius we are checking
        for (int r = minRadius; r < maxRadius; r++) {
          for (int t = gradDir.at<float>(y,x) - thetaErrorRange; t < gradDir.at<float>(y,x) + thetaErrorRange; t++) {

            int xValNeg = x - (int) (r * std::cos(t));
            int yValNeg= y - (int) (r * std::sin(t));
            if (xValNeg >= 0 && xValNeg < width && yValNeg >= 0 && yValNeg < height) {
              houghSpace[xValNeg][yValNeg][r]++;
            }

            int xValPos = x + (int) (r * std::cos(t));
            int yValPos = y + (int) (r * std::sin(t));
            if (xValPos >= 0 && xValPos < width && yValPos >= 0 && yValPos < height) {
              houghSpace[xValPos][yValPos][r]++;
            }
          }
        }
      }
    }
    std::cout << x << std::endl;
  }
  
  std::cout << "Hough space generated" << std::endl;

  // -- Covert the 3D houghSpace into a 2D image so we can have a look -- //
  if (drawHoughSpace) {
    cv::Mat houghSpaceImage = cv::Mat::zeros(width, height, CV_32SC1);
    for (int x = 0; x < width; x++){
      for (int y = 0; y < height; y++){
        for (int r = minRadius; r < maxRadius; r++){
        	houghSpaceImage.at<int>(y,x) += houghSpace[x][y][r];
        }
      }
    }
    cv::imwrite("cirlce-hough-space.jpg", houghSpaceImage);
  }

  // -- Extract cicles -- // 
  std::vector<cv::Vec3i> circles;
 
  std::cout << "Getting peaks" << std::endl;
  // Loop until there are no more values in the hough space greater than the threshold
  while (true) {

    // Find the largest value in the hough space
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
      std::cout << "Got peak: " << currentMax << std::endl;
    }
    else {
      // If the max value is less than the threashold stop the loop 
      break;
    }

  }
  std::cout << "Got peaks" << std::endl;
    
  cv::Mat circlesOutput;
  input.copyTo(circlesOutput);
  draw(circles, circlesOutput, cv::Scalar(0, 0, 0));
  std::cout << "Circles length:" << circles.size() << std::endl;
  cv::imwrite("cirlce-hough-output.jpg", circlesOutput);

  return circles;
}


std::vector<cv::Vec2f> houghLines(cv::Mat &input) {
  
  cv::Mat gradMag = gradientMagnitude(input);
  cv::Mat gradDir = gradientDirection(input);
  
  cv::imwrite("lines-grad-mag.png", gradMag);
  cv::imwrite("lines-input.png", input);
  
  double width = input.size().width, height = input.size().height;
  double imageHypot = std::hypot(width, height);

  int numberOfRadii = 2*(width + height);
  int numberOfAngles = 360;

  int gradientThreshold = 180;
  int thetaRange = 180;
  int houghThreshold = 76;

  cv::Mat houghSpace = cv::Mat::zeros(numberOfRadii, numberOfAngles, CV_32FC1);
  std::cout << "Hough space height: " << houghSpace.size().height << std::endl;
  std::cout << "Hough space width: " << houghSpace.size().width << std::endl;

  // For every pixel in the image 
  for (int y = 0; y < gradMag.rows; y++) {
    for (int x = 0; x < gradMag.cols; x++) {
      
      // If the pixel is above gradient mag threshold (i.e on an edge)
      if (gradMag.at<float>(y, x) > gradientThreshold) {

        for (int theta = -thetaRange; theta < thetaRange; theta++) {
          int rho = x * std::cos(theta * (CV_PI / (double)180)) + y * std::sin(theta * (CV_PI / (double)180)) + width + height;
          //std::cout << rho << ", " << theta << std::endl;
          houghSpace.at<float>(rho, theta)++;
        }
      }
    }
  }

  std::cout << "Got hough space" << std::endl;
  cv::imwrite("lines-hough-space.png", houghSpace);
  std::cout << "Drew hough space" << std::endl;

  std::vector<cv::Vec2f> houghLines;
  
  // Loop until there are no more values in the hough space greater than the threshold
  while (true) {

    // Find the largest value in the hough space
    int currentMax = 0;
    int maxTheta = 0;
    int maxRho = 0;

    for (int rho = 0; rho < numberOfRadii; rho++) {
      for (int theta = 0; theta < numberOfAngles; theta++) {
        if (houghSpace.at<float>(rho, theta) > currentMax) {
          currentMax = houghSpace.at<float>(rho,theta);
          maxRho = rho;
          maxTheta = theta;
        }
      }
    }

    // Black out pixles around the peak to avoid clustering
    if (currentMax > houghThreshold) {
      cv::Vec2f lineAtMax = cv::Vec2f(maxTheta * (CV_PI/180), maxRho - width - height);
      houghLines.push_back(lineAtMax);
      int blackOutRadius = 30;
      for ( float theta = maxTheta - blackOutRadius; theta <= maxTheta + blackOutRadius; theta++) {
        for ( float rho = maxRho - blackOutRadius; rho <= maxRho + blackOutRadius; rho++) {
          if (theta >= 0 && rho>=0 && rho < numberOfRadii && theta < numberOfAngles) {
            houghSpace.at<float>(rho, theta) = 0.0f;
          }
        }
      }
      std::cout << currentMax << std::endl;
    }

    else {
      // If the max value is less than the threashold stop the loop 
      break;
    }

  }

  std::cout << "Got lines" << std::endl;
  std::cout << houghLines.size() << std::endl;

  cv::Mat output;
  input.copyTo(output);
  for( size_t i = 0; i < houghLines.size(); i++ )
  {
    std::pair<cv::Point, cv::Point> points = getTwoPointsOnLine(houghLines[i]);
    line( output, points.first, points.second, cv::Scalar(0,255,255), 2, 8);
  }
  cv::imwrite("myhough-output.png", output);

  return houghLines;
}

