#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <iostream>
#include <vector>
#include <array>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

class Image {
public:
	enum class DetectorType {
		SIFT,
		SURF,
		AKAZE
	};

	void loadImage(const std::string& kImagePath);
	void detectKeyPoints(DetectorType type);
	const std::vector<cv::KeyPoint>& getKeypoints();
	const cv::Mat& getDescriptor();

private:
	cv::Mat image_;
	std::vector<cv::KeyPoint> keypoints_;
	cv::Mat descriptor_;

	void detectKeyPoints(const cv::Mat& kImage, std::vector<cv::KeyPoint>& keypoint, cv::Mat& descriptor, DetectorType type) const;
};

#endif