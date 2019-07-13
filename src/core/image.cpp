#include "image.hpp"
#include "fileUtil.hpp"

Image::Image() {
	isRecoveredExtrinsicParameter_ = false;
	radial_distortion_[0] = 0.0;
	radial_distortion_[1] = 0.0;
}

void Image::loadAndDetectKeypoints(const std::string & kImagePath, DetectorType type) {
	// Load image infomation
	const cv::Mat kImage = cv::imread(kImagePath);
	image_size_.width = kImage.cols;
	image_size_.height = kImage.rows;
	principal_point_ = cv::Point2d(image_size_.width/2.0, image_size_.height/2.0);

	// Detect keypoint
	detectKeyPoints(kImage, keypoints_, descriptor_, type);
	for (int i = 0; i < static_cast<int>(keypoints_.size()); i++) {
		colors_.push_back(getPixelColor(kImage, i));
	}
	std::cout << keypoints_.size() << " keypoints were found." << std::endl;
}

const std::vector<cv::KeyPoint>& Image::getKeypoints() const {
	return keypoints_;
}

cv::Size2i Image::getImageSize() const {
	return image_size_;
}

const cv::Mat & Image::getDescriptor() const {
	return descriptor_;
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

cv::Vec3b Image::getKeypointColor(int keypoint_index) const {
	return colors_.at(keypoint_index);
}

void Image::setFileName(const std::string& file_name) {
	file_name_ = file_name;
}

void Image::writeImageInfo(const std::string& dir_path) const {
	const std::string kFilePath = FileUtil::addSlashToLast(dir_path) + file_name_ + ".image";
	cv::FileStorage fs(kFilePath, cv::FileStorage::WRITE);
	cv::write(fs, "keypoints", keypoints_);
	cv::write(fs, "descriptors", descriptor_);
	cv::write(fs, "colors", colors_);
	const cv::Mat kImageSize = (cv::Mat_<int>(2, 1) << image_size_.width, image_size_.height);
	cv::write(fs, "size", kImageSize);

	fs.release();
}

void Image::loadImageInfo(const std::string & file_path) {
	cv::FileStorage fs(file_path, cv::FileStorage::READ);
	const cv::FileNode kKeypointsNode = fs["keypoints"];
	cv::read(kKeypointsNode, keypoints_);
	const cv::FileNode kDescriptorsNode = fs["descriptors"];
	cv::read(kDescriptorsNode, descriptor_);
	const cv::FileNode kColors = fs["colors"];
	cv::read(kColors, colors_);
	const cv::FileNode kImageSize = fs["size"];
	cv::Mat kImageSizeMat;
	cv::read(kImageSize, kImageSizeMat);
	image_size_.width = kImageSizeMat.at<int>(0, 0);
	image_size_.height = kImageSizeMat.at<int>(1, 0);
	principal_point_ = cv::Point2d(image_size_.width/2.0, image_size_.height/2.0);

	fs.release();
}

cv::Vec3b Image::getPixelColor(const cv::Mat& kImage, int x, int y) const {
	return kImage.at<cv::Vec3b>(y, x);
}

cv::Vec3b Image::getPixelColor(const cv::Mat& kImage, int keypoint_index) const {
	return getPixelColor(kImage, static_cast<int>(keypoints_.at(keypoint_index).pt.x), static_cast<int>(keypoints_.at(keypoint_index).pt.y));
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
