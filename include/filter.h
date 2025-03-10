/**
 * Rishi Patel
 * Due: 01/23/2025
 * 
 * This purpose of this file is to implement the filter functions.
 * 
 */

#ifndef FILTER_H
#define FILTER_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// alternativeGrayscale
int alternativeGrayscale(cv::Mat &src, cv::Mat &dst);

// sepia filter
int sepia_filter(cv::Mat &src, cv::Mat &dst);

// blur5x5_1
int blur5x5_1(cv::Mat &src, cv::Mat &dst);

// blur5x5_2
int blur5x5_2(cv::Mat &src, cv::Mat &dst);

// Sobel filter functions
int sobelX3x3(cv::Mat &src, cv::Mat &dst);
int sobelY3x3(cv::Mat &src, cv::Mat &dst);

// Gradient magnitude function
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);

// blurQuantize
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);

// cartoon_filter
int cartoon_filter(cv::Mat &src, cv::Mat &dst);

// sketch_filter
int sketch_filter(cv::Mat &src, cv::Mat &dst);

// alternativeGrayscale3
int alternativeGrayscale3(cv::Mat &src, cv::Mat &dst);  

#endif // FILTER_H
