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

#include "hough.h"
#include "utils.h"


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
  std::vector<cv::Vec3i> hough_circles = houghCircles(image_gray, 30);
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
    cv::imwrite("graddirection.png", gradD * 100);
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

