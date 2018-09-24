#include "image.hpp"

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

cv::Matx33d Image::getIntrinsicParameter() const {
	return cv::Matx33d(focal_length_, 0.0, principal_point_.x
					, 0.0, focal_length_, principal_point_.y
					, 0.0, 0.0, 1.0);
}

void Image::setExtrinsicParameter(const cv::Matx33d & rotation_mat, const cv::Matx31d & translation_vec) {
	rotation_mat_ = rotation_mat;
	translation_vec_ = translation_vec;
}

cv::Vec3b Image::getPixelColor(int x, int y) const {
	return image_.at<cv::Vec3b>(y, x);
}

cv::Vec3b Image::getPixelColor(int keypoint_index) const {
	return getPixelColor(keypoints_.at(keypoint_index).pt.x, keypoints_.at(keypoint_index).pt.y);
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

	//cv::Mat dst_img;
	//cv::drawKeypoints(kImage, keypoint, dst_img);
	//cv::imshow("", dst_img);
	//cv::waitKey(0);
}
