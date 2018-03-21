#include "image.hpp"

void Image::loadImage(const std::string & kImagePath) {
	image_ = cv::imread(kImagePath);
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
