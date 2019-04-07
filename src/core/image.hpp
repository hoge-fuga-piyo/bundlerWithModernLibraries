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

	Image();
	void loadImage(const std::string& kImagePath);
	void detectKeyPoints(DetectorType type);
	bool isRecoveredExtrinsicParameter() const;

public: // Setter/getter
	const std::vector<cv::KeyPoint>& getKeypoints() const;
	const cv::Mat& getDescriptor() const;
	const cv::Mat& getImage() const;
	void setFocalLength(double focal_length);
	void setPrincipalPoint(double cx, double cy);
	cv::Matx33d getIntrinsicParameter() const;
	cv::Vec3d getRotationAngleAxis() const;
	cv::Matx33d getRotationMatrix() const;
	cv::Matx31d getTranslation() const;
	cv::Matx34d getExtrinsicParameter() const;
	cv::Matx34d getProjectionMatrix() const;
	std::array<double, 2> getRadialDistortion() const;
	void setExtrinsicParameter(const cv::Matx33d& rotation_mat, const cv::Matx31d& translation_vec);
	cv::Vec3b getPixelColor(int x, int y) const;
	cv::Vec3b getPixelColor(int keypoint_index) const;

private: // Instance variables
	cv::Mat image_;
	std::vector<cv::KeyPoint> keypoints_;
	cv::Mat descriptor_;
	double focal_length_;
	cv::Point2d principal_point_;
	cv::Matx33d rotation_mat_;
	cv::Matx31d translation_vec_;
	std::array<double, 2> radial_distortion_;
	bool isRecoveredExtrinsicParameter_;

private: // Methods
	void detectKeyPoints(const cv::Mat& kImage, std::vector<cv::KeyPoint>& keypoint, cv::Mat& descriptor, DetectorType type) const;
};

#endif