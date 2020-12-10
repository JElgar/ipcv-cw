#include "utils.h"

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

void draw(cv::Rect rect, cv::Mat frame, cv::Scalar colour) {
    rectangle(frame, rect, colour, 2);
}

void draw(std::vector<cv::Rect> rects, cv::Mat frame, cv::Scalar colour) {
  for ( int i = 0; i < rects.size(); i++ ) {
    draw(rects[i], frame, colour);
  }
}

void draw(std::vector<cv::Vec3i> circles, cv::Mat frame, cv::Scalar colour) {
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

cv::Rect circleToRect(cv::Vec3i circle) {
  return cv::Rect(cv::Point(circle[0] - circle[2], circle[1] - circle[2]), cv::Point(circle[0] + circle[2], circle[1] + circle[2]));
}
