#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <sstream>
#include <fstream>
#include <string>
#include <vector>
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

std::pair<cv::Point, cv::Point> getTwoPointsOnLine(cv::Vec2f line) {
  float rho = line[1];
  float theta = line[0];
  double a = cos(theta), b = sin(theta);
  double x0 = a*rho, y0 = b*rho;
  cv::Point point1 = cv::Point(cvRound(x0 + 1000*(-b)), cvRound(y0 + 1000*(a)));
  cv::Point point2 = cv::Point(cvRound(x0 - 1000*(-b)), cvRound(y0 - 1000*(a)));
  return std::pair<cv::Point, cv::Point>(point1, point2);
}

cv::Rect circleToRect(cv::Vec3i circle) {
  return cv::Rect(cv::Point(circle[0] - circle[2], circle[1] - circle[2]), cv::Point(circle[0] + circle[2], circle[1] + circle[2]));
}

void draw(cv::Rect rect, cv::Mat frame, cv::Scalar colour) {
    rectangle(frame, rect, colour, 2);
}

void draw(std::vector<cv::Rect> rects, cv::Mat frame, cv::Scalar colour) {
  for ( int i = 0; i < rects.size(); i++ ) {
    draw(rects[i], frame, colour);
  }
}

void draw(std::vector<cv::Vec3i> circles, cv::Mat frame, cv::Scalar colour = cv::Scalar(255, 0, 255)) {
  for (cv::Vec3i circle : circles) {
    std::cout << circle[2] << std::endl;
    cv::Rect circle_rect = circleToRect(circle);
    cv::Point center = cv::Point(circle[0], circle[1]);
    // circle center
    cv::circle( frame , center, 1, cv::Scalar(0,100,100), 3, 8);
    // circle outline
    int radius = circle[2];
    cv::circle( frame , center, radius, colour, 3, 8);
    cv::rectangle(frame, circle_rect , colour, 2);
  }
}


cv::Point lineIntersection(cv::Vec2f line1, cv::Vec2f line2) {
  float theta1 = line1[0];
  float rho1 = line1[1];
  float theta2 = line2[0];
  float rho2 = line2[1];
  float determinent = std::cos(theta1) *  std::sin(theta2) - cv::sin(theta1) * std::cos(theta2);

  // Lines parallel therefore no intersection 
  if (determinent == 0) {
    return cv::Point();
  }

  else {
    int x = (cv::sin(theta2) * rho1 - cv::sin(theta1) * rho2) / determinent;
    int y = (-cv::cos(theta2) * rho1 + cv::cos(theta1) * rho2) / determinent;
    return cv::Point(x, y);
  }
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


std::vector<cv::Vec3i> houghCircles (cv::Mat &input, int threshold = 14, bool drawHoughSpace = false) {
  
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
  draw(circles, circlesOutput);
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

float intersection_over_union(cv::Rect detected_rect, cv::Rect true_rect) {
  return (detected_rect & true_rect).area() / (float)(detected_rect | true_rect).area();
}

float intersection_over_union(cv::Vec3i circle, cv::Rect rect) {
  cv::Rect circle_rect = circleToRect(circle);
  return std::max(intersection_over_union(circle_rect, rect), intersection_over_union(rect, circle_rect));
}


int number_of_correctly_detected_faces(std::vector<cv::Rect> detected_rects, std::vector<cv::Rect> true_rects, float threshold = 0.5) {
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
      std::cout << "Image at index " << i << " rejected with IOIU " << max_iou << std::endl;
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

std::vector<cv::Rect> violaJonesDartDetection(cv::Mat &input) {
  cv::String cascade_name = "dartcascade/cascade.xml";
  cv::CascadeClassifier cascade;
  if( !cascade.load( cascade_name ) ){ 
    printf("--(!)Error loading\n"); 
    std::vector<cv::Rect> rects;
    return rects;
  }
    
  std::vector<cv::Rect> faces;
  cascade.detectMultiScale( input, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, cv::Size(50, 50), cv::Size(500,500) );
    
  cv::Mat vjOutput;
  input.copyTo(vjOutput);
  draw(faces, vjOutput, cv::Scalar(0, 255, 0));
  cv::imwrite("vj-output.jpg", vjOutput);
  return faces;
}

std::vector<cv::Rect> detectDartboards(cv::Mat image_gray, float threshold = 0.15) {

  // Do vj and hough and store results
  std::vector<cv::Vec3i> hough_circles = houghCircles(image_gray, 28);
  std::vector<cv::Rect> vj_boards = violaJonesDartDetection(image_gray);
  std::vector<cv::Vec2f> hough_lines = houghLines(image_gray);

  std::vector<int> boardsDetectedInStage1;
  std::vector<cv::Rect> combined_boards; 

  // Find IOU of hough_boards and vj_boards and  
  for (int i = 0; i < vj_boards.size(); i++) {
    cv::Rect vj_board = vj_boards[i];
    float max_iou = 0;
    cv::Vec3i max_hough_board;
    for (cv::Vec3i h_board : hough_circles) {
      float iou = intersection_over_union(h_board, vj_board);
      //cout << "IOU: "<< iou << endl;
      if (iou > max_iou) {
        max_iou = iou;
        max_hough_board = h_board; 
      }
    }
    std::cout << "The max_iou in thing is: " << max_iou << std::endl;
    if (max_iou > threshold) {

      // If this circle is an inner circle of another circle then lets take the outer circle
      // The helps with the consentric rings of the dart board 
      int centerPointThreshold = 20;
      for (cv::Vec3i circle : hough_circles) {
        if (circle != max_hough_board) {
          cv::Point distance = cv::Point(circle[0], circle[1]) - cv::Point(max_hough_board[0], max_hough_board[1]);
          if (cv::sqrt(distance.x*distance.x + distance.y*distance.y) < centerPointThreshold) {
            // If the current max has a smaller radius than this circle, set the current (larger) circle to the max 
            if (max_hough_board[2] < circle[2]) {
              max_hough_board = circle;
            }
          }
        }
      }
      // Push the thing with the max area
      combined_boards.push_back(circleToRect(max_hough_board));
      //if(circleToRect(max_hough_board).area() > vj_board.area()) {
      //} else {
      //  std::cout << "Took board with area " << vj_board.area() << " instead of circle " << circleToRect(max_hough_board) << std::endl;
      //  combined_boards.push_back(vj_board);
      //}

      boardsDetectedInStage1.push_back(i);
    } 
  }

  // Remove boards detected in stage 1
  for (int index : boardsDetectedInStage1) {
    vj_boards.erase(vj_boards.begin() + index);
  }

  // For the remaining boards
  // If 3 lines pass through the same point (with threshold) and that point is in the bounding box
  //  \|
  // __\__
  //   |\
  //   | \

  std::vector<cv::Point> lineIntersectionPoints;
  for (cv::Vec2f line1 : hough_lines) {
    for (cv::Vec2f line2 : hough_lines) {
      if (line1 != line2) {
        cv::Point point = lineIntersection(line1, line2);
        std::cout << point.x << ", " << point.y << std::endl;
        lineIntersectionPoints.push_back(point);
      }
    }
  }
    
  int intersectionThreshold = 2;
  int requiredIntersections = 3;

  for(cv::Rect vj_board : vj_boards) {

    int numberOfIntersectionsFound = 2;
  
    std::vector<cv::Point> currentBoardsIntersectionPoints;
    for (cv::Point point : lineIntersectionPoints) {
      if ( vj_board.contains(point) ) {
        currentBoardsIntersectionPoints.push_back(point);
      }
    }

    for (cv::Point point1 : currentBoardsIntersectionPoints) {
      if (numberOfIntersectionsFound >= requiredIntersections) {
        break;
      }
      for (cv::Point point2 : currentBoardsIntersectionPoints) {
        if (point1 != point2) {
          // If the distance between this intersection point and the other intersection point is < the threshold
          cv::Point distance = point1 - point2;
          if (cv::sqrt(distance.x*distance.x + distance.y*distance.y) < intersectionThreshold) {
            combined_boards.push_back(vj_board);
            numberOfIntersectionsFound++;
            std::cout << "Interseciton found" << std::endl;
            break;
          }
        }
      }
    }
  }

  std::cout << "True outputs before filtering" << std::endl;

  // Remove duplicates
  std::vector<cv::Rect> combined_boards_without_duplicated;
  for (int i = 0; i < combined_boards.size(); i++) {
    bool foundDuplicate = false;
    for (int j = i - 1; j >= 0; j--) {
      if (combined_boards[i] == combined_boards[j]) {
        foundDuplicate = true;
      }
    }
    if (!foundDuplicate) {
      combined_boards_without_duplicated.push_back(combined_boards[i]);
    }
  }
      
  // If this circle is an inner circle of another circle then lets take the outer circle
  // The helps with the consentric rings of the dart board 
  //std::vector<int> boardsContainedByOtherBoards;
  //int centerPointThreshold = 20;
  //for (int i = 0; i < combined_boards_without_duplicated.size(); i++) {
  //  bool foundOuterCircle = false;
  //  for (int j = i - 1; j >= 0; j--) {
  //    cv::Point distance = cv::Point(circle[0], circle[1]) - cv::Point(max_hough_board[0], max_hough_board[1]);
  //    if (cv::sqrt(distance.x*distance.x + distance.y*distance.y) < centerPointThreshold) {
  //      // If the current max has a smaller radius than this circle, set the current (larger) circle to the max 
  //      if (max_hough_board[2] < circle[2]) {
  //        max_hough_board = circle;
  //      }
  //    }
  //  }
  //  if (!foundOuterCircle) {
  //    combined_boards_without_duplicated.push_back(combined_boards[i]);
  //  }
  //}
  return combined_boards_without_duplicated;
}


int
main (int argc, char **argv)
{

    cv::Mat image, image_gray;
    image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);

    cv::cvtColor( image, image_gray, CV_BGR2GRAY);
    
    // hough lines
    cv::imwrite("lines_input_before_input.png", image_gray);
    //houghLines(image_gray);
    //cv::medianBlur(image_gray, image_gray, 5);

    cv::Mat gradD = gradientDirection(image_gray);
    cv::Mat gradM = gradientMagnitude(image_gray);
    cv::imwrite("graddirection.png", gradD * 10);
    cv::imwrite("gradmag.png", gradM);

    std::vector<cv::Rect> true_faces;
    if (argc > 2) {
      true_faces = get_true_face(argv[2]);
    }
    

    // -- Hough -- //
    //std::vector<cv::Vec3i> circles = houghCircles(image_gray, 35);
    //cv::Mat circlesOutput;
    //image.copyTo(circlesOutput);
    //draw(circles, circlesOutput);
    //std::cout << "Circles length:" << circles.size() << std::endl;
    //cv::imwrite("cirlce-hough-output.jpg", circlesOutput);

    //std::cout << "Hough done" <<  std::endl;
   
    // -- Viola Jones -- //
    //std::vector<cv::Rect> boards = voilaJonesDartDetection(image_gray);
    //cv::Mat vjOutput;
    //image.copyTo(vjOutput);
    //draw(boards, vjOutput, cv::Scalar(0, 255, 0));
    //draw(true_faces, vjOutput, cv::Scalar(0, 0, 255));
    //std::cout << "Boards length:" << boards.size() << std::endl;
    //cv::imwrite("vj-output.jpg", vjOutput);

    std::vector<cv::Rect> finalBoards = detectDartboards(image_gray);
    cv::Mat finalOutput;
    image.copyTo(finalOutput);
    draw(finalBoards, finalOutput, cv::Scalar(0, 255, 0));
    cv::imwrite("final-output.jpg", finalOutput);

    // -- Compare with true values -- //
    std::cout << "Number of true faces: " <<  true_faces.size() << std::endl;
    std::cout << "NUmber of final detections: " <<  finalBoards.size() << std::endl;
    std::cout << "TPR: " << true_positive_rate(finalBoards, true_faces) << std::endl;
    std::cout << "F1: " << f1_score(finalBoards, true_faces) << std::endl;
   
    draw(true_faces, finalOutput, cv::Scalar(0, 0, 255));
    cv::imwrite("final-output-true.jpg", finalOutput);

    // free memory
    image.release();

    return 0;
}

