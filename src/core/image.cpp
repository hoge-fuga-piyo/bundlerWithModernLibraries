#include "image.hpp"

Image::Image() {
	isRecoveredExtrinsicParameter_ = false;
	radial_distortion_[0] = 0.0;
	radial_distortion_[1] = 0.0;
}

void Image::loadImage(const std::string & kImagePath) {
	image_ = cv::imread(kImagePath);
	principal_point_ = cv::Point2d(image_.cols/2.0, image_.rows/2.0);
}

void Image::detectKeyPoints(DetectorType type) {
	detectKeyPoints(image_, keypoints_, descriptor_, type);
}

const std::vector<cv::KeyPoint>& Image::getKeypoints() const {
	return keypoints_;
}

const cv::Mat & Image::getDescriptor() const {
	return descriptor_;
}

const cv::Mat & Image::getImage() const {
	return image_;
}

void Image::setFocalLength(double focal_length) {
	focal_length_ = focal_length;
}

void Image::setPrincipalPoint(double cx, double cy) {
	principal_point_.x = cx;
	principal_point_.y = cy;
}

cv::Matx33d Image::getIntrinsicParameter() const {
	return cv::Matx33d(focal_length_, 0.0, principal_point_.x
					, 0.0, focal_length_, principal_point_.y
					, 0.0, 0.0, 1.0);
}

cv::Vec3d Image::getRotationAngleAxis() const {
	cv::Vec3d rotation_vec;
	cv::Rodrigues(rotation_mat_, rotation_vec);
	return rotation_vec;
}

cv::Matx33d Image::getRotationMatrix() const {
	return rotation_mat_;
}

cv::Matx31d Image::getTranslation() const {
	return translation_vec_;
}

cv::Matx34d Image::getExtrinsicParameter() const {
	return cv::Matx34d(rotation_mat_(0, 0), rotation_mat_(0, 1), rotation_mat_(0, 2), translation_vec_(0)
		, rotation_mat_(1, 0), rotation_mat_(1, 1), rotation_mat_(1, 2), translation_vec_(1)
		, rotation_mat_(2, 0), rotation_mat_(2, 1), rotation_mat_(2, 2), translation_vec_(2));
}

cv::Matx34d Image::getProjectionMatrix() const {
	return getIntrinsicParameter() * getExtrinsicParameter();
}

std::array<double, 2> Image::getRadialDistortion() const {
	return radial_distortion_;
}

void Image::setExtrinsicParameter(const cv::Matx33d & rotation_mat, const cv::Matx31d & translation_vec) {
	rotation_mat_ = rotation_mat;
	translation_vec_ = translation_vec;
	isRecoveredExtrinsicParameter_ = true;
}

cv::Vec3b Image::getPixelColor(int x, int y) const {
	return image_.at<cv::Vec3b>(y, x);
}

cv::Vec3b Image::getPixelColor(int keypoint_index) const {
	return getPixelColor(static_cast<int>(keypoints_.at(keypoint_index).pt.x), static_cast<int>(keypoints_.at(keypoint_index).pt.y));
}

bool Image::isRecoveredExtrinsicParameter() const {
	return isRecoveredExtrinsicParameter_;
}

void Image::detectKeyPoints(const cv::Mat& kImage, std::vector<cv::KeyPoint>& keypoint, cv::Mat & descriptor, DetectorType type) const {
	if (type == DetectorType::SIFT) {
		cv::Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create();
		detector->detectAndCompute(kImage, cv::Mat(), keypoint, descriptor);
	}
	else if (type == DetectorType::SURF) {
		cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create();
		detector->detectAndCompute(kImage, cv::Mat(), keypoint, descriptor);
	}
	else if (type == DetectorType::AKAZE) {
		cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();
		detector->detectAndCompute(kImage, cv::Mat(), keypoint, descriptor);
	}
}
