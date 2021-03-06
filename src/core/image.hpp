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
	void loadAndDetectKeypoints(const std::string& kImagePath, DetectorType type);
	bool isRecoveredExtrinsicParameter() const;

public: // Setter/getter
	const std::vector<cv::KeyPoint>& getKeypoints() const;
	cv::Size2i getImageSize() const;
	const cv::Mat& getDescriptor() const;
	void setFocalLength(double focal_length);
	void setPrincipalPoint(double cx, double cy);
	cv::Matx33d getIntrinsicParameter() const;
	double getFocalLength() const;
	cv::Vec3d getRotationAngleAxis() const;
	cv::Matx33d getRotationMatrix() const;
	cv::Matx31d getTranslation() const;
	cv::Matx34d getExtrinsicParameter() const;
	cv::Matx34d getProjectionMatrix() const;
	std::array<double, 2> getRadialDistortion() const;
	void setExtrinsicParameter(const cv::Matx33d& kRotationMatrix, const cv::Matx31d& kTranslation);
	cv::Vec3b getKeypointColor(int keypoint_index) const;
	void setFileName(const std::string& kFileName);
	std::string getFileName() const;
	bool hasExifFocalLength() const;
	void writeImageInfo(const std::string& kDirPath) const;
	void loadImageInfo(const std::string& kFileName);

private: // Instance variables
	//cv::Mat image_;
	cv::Size2i image_size_;
	std::vector<cv::KeyPoint> keypoints_;
	std::vector<cv::Vec3b> colors_;
	cv::Mat descriptor_;
	double focal_length_;
	bool has_focal_length_;
	cv::Point2d principal_point_;
	cv::Matx33d rotation_mat_;
	cv::Matx31d translation_vec_;
	std::array<double, 2> radial_distortion_;
	bool isRecoveredExtrinsicParameter_;

	std::string file_name_;

private: // Methods
	//const cv::Mat& getImage() const;
	void detectKeyPoints(const cv::Mat& kImage, std::vector<cv::KeyPoint>& keypoint, cv::Mat& descriptor, DetectorType type) const;
	cv::Vec3b getPixelColor(const cv::Mat& kImage, int x, int y) const;
	cv::Vec3b getPixelColor(const cv::Mat& kImage, int keypoint_index) const;
};

#endif