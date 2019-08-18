#include "image.hpp"
#include "fileUtil.hpp"
#include "exifInfo.hpp"

Image::Image() {
	isRecoveredExtrinsicParameter_ = false;
	radial_distortion_[0] = 0.0;
	radial_distortion_[1] = 0.0;
	has_focal_length_ = false;
}

/**
 * @brief load image, extract focal length from exif, and extract keypoints
 * @param[in] kImagePath image file path
 * @param[in] type feature descriptor type such as SIFT
 */
void Image::loadAndDetectKeypoints(const std::string & kImagePath, DetectorType type) {
	// Load image infomation
	const cv::Mat kImage = cv::imread(kImagePath);
	image_size_.width = kImage.cols;
	image_size_.height = kImage.rows;
	principal_point_ = cv::Point2d(image_size_.width/2.0, image_size_.height/2.0);

	// Load exif
	ExifInfo exif_info;
	bool has_exif = exif_info.loadImage(kImagePath);
	if (has_exif && exif_info.hasFocalLengthInPixel()) {
		focal_length_ = exif_info.getFocalLengthInPixel();
		has_focal_length_ = true;
		std::cout << "Use exif focal length: " << focal_length_ << std::endl;
	} else {
		std::cout << "Use default focal length" << std::endl;
	}

	// Detect keypoint
	detectKeyPoints(kImage, keypoints_, descriptor_, type);
	for (int i = 0; i < static_cast<int>(keypoints_.size()); i++) {
		colors_.push_back(getPixelColor(kImage, i));
	}
	std::cout << keypoints_.size() << " keypoints were found." << std::endl;
}

/**
 * @brief get keypoints of image
 * @return keypoints
 */
const std::vector<cv::KeyPoint>& Image::getKeypoints() const {
	return keypoints_;
}

/**
 * @brief get image width and height
 * @return image width and height
 */
cv::Size2i Image::getImageSize() const {
	return image_size_;
}

/**
 * @brief get feature descriptor of each keypoint
 * @return feature descriptor
 */
const cv::Mat & Image::getDescriptor() const {
	return descriptor_;
}

/**
 * @brief set focal length expressed in pixel
 * @param[in] focal_length focal length expressed in pixel
 */
void Image::setFocalLength(double focal_length) {
	focal_length_ = focal_length;
}

/**
 * @brief set principal point
 * @param[in] cx principal point of x axis
 * @param[in] cy principal point of y axis
 */
void Image::setPrincipalPoint(double cx, double cy) {
	principal_point_.x = cx;
	principal_point_.y = cy;
}

/**
 * @brief get intrinsic parameter(focal length and principal point)
 * @return intrinsic parameter
 */
cv::Matx33d Image::getIntrinsicParameter() const {
	return cv::Matx33d(focal_length_, 0.0, principal_point_.x
					, 0.0, focal_length_, principal_point_.y
					, 0.0, 0.0, 1.0);
}

/**
 * @brief get focal length
 * @return focal length
 */
double Image::getFocalLength() const {
	return focal_length_;
}

/**
 * @brief get rotation expressed in angle axis
 * @return rotation expressed in angle axis
 */
cv::Vec3d Image::getRotationAngleAxis() const {
	cv::Vec3d rotation_vec;
	cv::Rodrigues(rotation_mat_, rotation_vec);
	return rotation_vec;
}

/**
 * @brief get rotation matrix
 * @return rotation matrix
 */
cv::Matx33d Image::getRotationMatrix() const {
	return rotation_mat_;
}

/**
 * @brief get translation vector
 * @return translation vector of extrinsic parameter
 */
cv::Matx31d Image::getTranslation() const {
	return translation_vec_;
}

/**
 * @brief get extrinsic parameter
 * @return extrinsic parameter
 */
cv::Matx34d Image::getExtrinsicParameter() const {
	return cv::Matx34d(rotation_mat_(0, 0), rotation_mat_(0, 1), rotation_mat_(0, 2), translation_vec_(0)
		, rotation_mat_(1, 0), rotation_mat_(1, 1), rotation_mat_(1, 2), translation_vec_(1)
		, rotation_mat_(2, 0), rotation_mat_(2, 1), rotation_mat_(2, 2), translation_vec_(2));
}

/**
 * @brief get projection matrix
 * @return projection matrix
 */
cv::Matx34d Image::getProjectionMatrix() const {
	return getIntrinsicParameter() * getExtrinsicParameter();
}

std::array<double, 2> Image::getRadialDistortion() const {
	return radial_distortion_;
}

/**
 * @brief set extrinsic parameter
 * @param[in] kRotationMatrix rotation matrix
 * @param[in] kTranslation translation vector which 4-th col of extrinsic parameter
 */
void Image::setExtrinsicParameter(const cv::Matx33d & kRotationMatrix, const cv::Matx31d & kTranslation) {
	rotation_mat_ = kRotationMatrix;
	translation_vec_ = kTranslation;
	isRecoveredExtrinsicParameter_ = true;
}

cv::Vec3b Image::getKeypointColor(int keypoint_index) const {
	return colors_.at(keypoint_index);
}

/**
 * @brief set image file name
 * @param[in] kFileName image file name
 */
void Image::setFileName(const std::string& kFileName) {
	file_name_ = kFileName;
}

/**
 * @brief get image file name
 * @return image file name
 */
std::string Image::getFileName() const {
	return file_name_;
}

/**
 * @brief check image has exif focal length
 * @return return true if image has exif focal length
 */
bool Image::hasExifFocalLength() const {
	return has_focal_length_;
}

/**
 * @brief write image information to yaml file
 * @param[in] kDirPath directory path
 */
void Image::writeImageInfo(const std::string& kDirPath) const {
	const std::string kFilePath = FileUtil::addSlashToLast(kDirPath) + file_name_ + ".image";
	cv::FileStorage fs(kFilePath, cv::FileStorage::WRITE);
	cv::write(fs, "keypoints", keypoints_);
	cv::write(fs, "descriptors", descriptor_);
	cv::write(fs, "colors", colors_);
	const cv::Mat kImageSize = (cv::Mat_<int>(2, 1) << image_size_.width, image_size_.height);
	cv::write(fs, "size", kImageSize);
	const cv::Mat kIntrinsicParameter = (cv::Mat_<double>(3, 1) << focal_length_, principal_point_.x, principal_point_.y);
	cv::write(fs, "intrinsic", kIntrinsicParameter);
	cv::write(fs, "is_exif_focal_length", has_focal_length_);
	cv::write(fs, "image_name", file_name_);

	fs.release();
}

/**
 * @brief load image information
 * @param[in] kFileName file
 */
void Image::loadImageInfo(const std::string & kFileName) {
	cv::FileStorage fs(kFileName, cv::FileStorage::READ);
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
	const cv::FileNode kIntrinsicParameterNode = fs["intrinsic"];
	cv::Mat intrinsic_parameter;
	cv::read(kIntrinsicParameterNode, intrinsic_parameter);
	focal_length_ = intrinsic_parameter.at<double>(0, 0);
	principal_point_ = cv::Point2d(intrinsic_parameter.at<double>(1, 0), intrinsic_parameter.at<double>(2, 0));
	const cv::FileNode kHasFocalLengthNode = fs["is_exif_focal_length"];
	cv::read(kHasFocalLengthNode, has_focal_length_, false);
	const cv::FileNode kFileNameNode = fs["image_name"];
	cv::read(kFileNameNode, file_name_, "");

	fs.release();
}

/**
 * @brief get pixel color of a image
 * @param[in] kImage image
 * @param[in] x x coordinate on the image
 * @param[in] y y coordinate on the imge
 * @return pixel color
 */
cv::Vec3b Image::getPixelColor(const cv::Mat& kImage, int x, int y) const {
	return kImage.at<cv::Vec3b>(y, x);
}

/**
 * @brief get pixel color of a image
 * @param[in] kImage image
 * @param[in] keypoint_index keypoint index
 * @return pixel color
 */
cv::Vec3b Image::getPixelColor(const cv::Mat& kImage, int keypoint_index) const {
	return getPixelColor(kImage, static_cast<int>(keypoints_.at(keypoint_index).pt.x), static_cast<int>(keypoints_.at(keypoint_index).pt.y));
}

/**
 * @brief check extrinsic parameter already be recovered
 * @return return true if extrinsic parameter was recovered
 */
bool Image::isRecoveredExtrinsicParameter() const {
	return isRecoveredExtrinsicParameter_;
}

/**
 * @brief detect keypoints from the image
 * @param[in] kImage image
 * @param[out] keypoint keypoints
 * @param[out] descriptor feature descriptors
 * @param[in] type type of feature point extraction
 */
void Image::detectKeyPoints(const cv::Mat& kImage, std::vector<cv::KeyPoint>& keypoint, cv::Mat & descriptor, DetectorType type) const {
	cv::UMat umat_image;
	kImage.copyTo(umat_image);
	if (type == DetectorType::SIFT) {
		cv::Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create();
		detector->detectAndCompute(umat_image, cv::Mat(), keypoint, descriptor);
	}
	else if (type == DetectorType::SURF) {
		cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create();
		detector->detectAndCompute(umat_image, cv::Mat(), keypoint, descriptor);
	}
	else if (type == DetectorType::AKAZE) {
		cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();
		detector->detectAndCompute(umat_image, cv::Mat(), keypoint, descriptor);
	}
}
