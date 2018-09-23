#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <iostream>
#include <vector>
#include <array>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

class Image {
public: // Methods
	enum class DetectorType {
		SIFT,
		SURF,
		AKAZE
	};

	void loadImage(const std::string& kImagePath);
	void detectKeyPoints(DetectorType type);

public: // Setter/getter
	const std::vector<cv::KeyPoint>& getKeypoints() const;
	const cv::Mat& getDescriptor() const;
	const cv::Mat& getImage() const;
	void setFocalLength(double focal_length);
	cv::Matx33d getIntrinsicParameter() const;
	void setExtrinsicParameter(const cv::Matx33d& rotation_mat, const cv::Matx31d& translation_vec);

private: // Instance variables
	cv::Mat image_;
	std::vector<cv::KeyPoint> keypoints_;
	cv::Mat descriptor_;
	double focal_length_;
	cv::Point2f principal_point_;
	cv::Matx33d rotation_mat_;
	cv::Matx31d translation_vec_;

private: // Methods
	void detectKeyPoints(const cv::Mat& kImage, std::vector<cv::KeyPoint>& keypoint, cv::Mat& descriptor, DetectorType type) const;
};

#endif