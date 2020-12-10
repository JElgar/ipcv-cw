#pragma once
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

std::vector<std::string> split(std::string line, std::string delimiter);

std::pair<cv::Point, cv::Point> getTwoPointsOnLine(cv::Vec2f line);
void draw(cv::Rect rect, cv::Mat frame, cv::Scalar colour);
void draw(std::vector<cv::Rect> rects, cv::Mat frame, cv::Scalar colour);
void draw(std::vector<cv::Vec3i> circles, cv::Mat frame, cv::Scalar colour = cv::Scalar(255, 0, 255));
cv::Point lineIntersection(cv::Vec2f line1, cv::Vec2f line2);
cv::Rect circleToRect(cv::Vec3i circle);

