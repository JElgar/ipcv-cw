#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "utils.h"

cv::Mat imageDx(cv::Mat &input);
cv::Mat imageDy(cv::Mat &input);
cv::Mat gradientMagnitude (cv::Mat &input);
cv::Mat gradientDirection (cv::Mat &input);
cv::Mat convolution(cv::Mat &input, cv::Mat &kernel);

// Hough
std::vector<cv::Vec3i> houghCircles (cv::Mat &input, int threshold = 30, bool drawHoughSpace = false);
std::vector<cv::Vec2f> houghLines(cv::Mat &input);
