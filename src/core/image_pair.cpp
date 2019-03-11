#include "image_pair.hpp"
#include "cvUtil.hpp"
#include "mathUtil.hpp"
#include <opencv2/viz.hpp>

const double ImagePair::kDistanceRatioThreshold_ = 0.6;

ImagePair::ImagePair(){}

/**
 * @brief	Keypoint matching between kImage1 and kImage2.
 * @param	kImage1		Image 1
 * @param	kImage2		Image 2
 */
void ImagePair::keypointMatching(const Image &kImage1, const Image &kImage2) {
	const cv::Mat& kImg1 = kImage1.getImage();
	const cv::Mat& kImg2 = kImage2.getImage();
	const std::vector<cv::KeyPoint>& kKeypoints1 = kImage1.getKeypoints();
	const std::vector<cv::KeyPoint>& kKeypoints2 = kImage2.getKeypoints();
	const cv::Mat& kDescriptor1 = kImage1.getDescriptor();
	const cv::Mat& kDescriptor2 = kImage2.getDescriptor();

	matches_ = keypointMatching(kImg1, kKeypoints1, kDescriptor1, kImg2, kKeypoints2, kDescriptor2);
}

std::vector<cv::DMatch> ImagePair::keypointMatching(const cv::Mat& kImage1, const std::vector<cv::KeyPoint>& kKeypoints1, const cv::Mat& kDescriptor1
	, const cv::Mat& kImage2, const std::vector<cv::KeyPoint>& kKeypoints2, const cv::Mat& kDescriptor2){

	// keypoint matching
	cv::FlannBasedMatcher matcher;
	std::vector<std::vector<cv::DMatch>> matches12, matches21;
	matcher.knnMatch(kDescriptor1, kDescriptor2, matches12, 2);
	matcher.knnMatch(kDescriptor2, kDescriptor1, matches21, 2);

	// cross check
	crossCheck(matches12, matches21);
	std::cout << "Keypoint num after cross check: " << matches12.size() << std::endl;

	// remove wrong matching by using distance on feature point space
	std::vector<bool> is_good_matches = findGoodKeypointMatchingByDistanceRatio(matches12, matches21, kDistanceRatioThreshold_);
	std::vector<cv::DMatch> good_matches = removeWrongKeypointMatching(matches12, is_good_matches);
	std::cout << "Keypoint num after distance condtion: " << good_matches.size() << std::endl;

	// remove wrong matching by using epipolar geometry
	cv::Mat is_good_matches_mat;
	findFundamentalMatrix(kImage1.size(), kKeypoints1, kImage2.size(), kKeypoints2, good_matches, is_good_matches_mat);
	good_matches = removeWrongKeypointMatching(good_matches, is_good_matches_mat);
	std::cout << "Keypoint num after epipolar: " << good_matches.size() << std::endl;

	return std::move(good_matches);
}

void ImagePair::showMatches(const cv::Mat & kImg1, const std::vector<cv::KeyPoint>& kKeypoints1, const cv::Mat & kImg2, const std::vector<cv::KeyPoint>& kKeypoints2) const {
	cv::Mat dst_img;
	cv::drawMatches(kImg1, kKeypoints1, kImg2, kKeypoints2, matches_, dst_img);
	cv::imshow("", dst_img);
	cv::waitKey(0);
}

double ImagePair::computeBaeslinePossibility(const Image & kImage1, const Image & kImage2, double homography_threshold) const {
	const std::vector<cv::KeyPoint>& kKeypoints1 = kImage1.getKeypoints();
	const std::vector<cv::KeyPoint>& kKeypoints2 = kImage2.getKeypoints();
	const cv::Size2i kImageSize1 = kImage1.getImage().size();
	const cv::Size2i kImageSize2 = kImage2.getImage().size();
	const int kMaxSize = std::max({kImageSize1.height, kImageSize1.width, kImageSize2.height, kImageSize2.width});

	return computeBaeslinePossibility(kKeypoints1, kKeypoints2, homography_threshold);
}

void ImagePair::recoverStructureAndMotion(const Image& kImage1, const Image& kImage2) {
	if (matches_.size() < 5) {
		std::cout << "[ERROR] The number of point correspondences is less than 5." << std::endl;
		return;
	}
	triangulated_points_.clear();

	const std::vector<cv::KeyPoint>& kKeypoints1 = kImage1.getKeypoints();
	const std::vector<cv::KeyPoint>& kKeypoints2 = kImage2.getKeypoints();
	const cv::Matx33d kIntrinsicParameter1 = kImage1.getIntrinsicParameter();
	const cv::Matx33d kIntrinsicParameter2 = kImage2.getIntrinsicParameter();

	std::vector<cv::Point2d> camera_vec1(matches_.size());
	std::vector<cv::Point2d> camera_vec2(matches_.size());
	for (size_t i = 0; i < matches_.size(); i++) {
		const cv::Point3d kVec1 = CvUtil::convertImagePointToCameraVector(kKeypoints1[matches_[i].queryIdx].pt, kIntrinsicParameter1);
		const cv::Point3d kVec2 = CvUtil::convertImagePointToCameraVector(kKeypoints2[matches_[i].trainIdx].pt, kIntrinsicParameter2);
		camera_vec1[i] = cv::Point2d(kVec1.x, kVec1.y);
		camera_vec2[i] = cv::Point2d(kVec2.x, kVec2.y);
	}
	
	const double kMinFocalLength = std::min((kIntrinsicParameter1(0, 0) + kIntrinsicParameter1(1, 1)) / 2.0, (kIntrinsicParameter2(0, 0) + kIntrinsicParameter2(1, 1)) / 2.0);
	const cv::Mat kEssentialMat = cv::findEssentialMat(camera_vec1, camera_vec2, cv::Matx33d::eye(), cv::RANSAC, 0.999, 1.0 / kMinFocalLength);
	cv::Mat triangulated_points;
	// Recover extrinsic parameter and triangulated 3d points. The points that has angle under 1 degree are assumed infinity points.
	cv::recoverPose(kEssentialMat, camera_vec1, camera_vec2, cv::Matx33d::eye(), rotation_mat_, translation_vec_, std::tan(MathUtil::convertDegreeToRadian(89.0)), cv::noArray(), triangulated_points);

	triangulated_points_.resize(triangulated_points.cols);
	for (int i = 0; i < triangulated_points.cols; i++) {
		cv::Mat homogeneous_point = triangulated_points.col(i) / triangulated_points.col(i).at<double>(3, 0);
		triangulated_points_[i] = cv::Point3d(homogeneous_point.at<double>(0, 0), homogeneous_point.at<double>(1, 0), homogeneous_point.at<double>(2, 0));
	}
}

const cv::Matx33d& ImagePair::getRotationMat() const {
	return rotation_mat_;
}

const cv::Matx31d& ImagePair::getTranslation() const {
	return translation_vec_;
}

double ImagePair::computeBaeslinePossibility(const std::vector<cv::KeyPoint>& kKeypoints1, const std::vector<cv::KeyPoint>& kKeypoints2, double homography_threshold) const {
	if (matches_.size() == 0) {
		return 0.0;
	}
	std::vector<int> good_keypoint_indexes1(matches_.size());
	std::vector<int> good_keypoint_indexes2(matches_.size());
	for (int i = 0; i < (int)matches_.size(); i++) {
		good_keypoint_indexes1[i] = matches_[i].queryIdx;
		good_keypoint_indexes2[i] = matches_[i].trainIdx;
	}

	std::vector<cv::Point2f> good_keypoints2f1;
	std::vector<cv::Point2f> good_keypoints2f2;
	cv::KeyPoint::convert(kKeypoints1, good_keypoints2f1, good_keypoint_indexes1);
	cv::KeyPoint::convert(kKeypoints2, good_keypoints2f2, good_keypoint_indexes2);

	std::vector<uchar> output_mask;
	cv::Matx33d homography_matrix = cv::findHomography(good_keypoints2f1, good_keypoints2f2, cv::RANSAC, homography_threshold, output_mask);

	int valid_count = 0;
	for (uchar mask : output_mask) {
		if (mask == 0) {
			valid_count++;
		}
	}

	return (double)valid_count / (double)matches_.size();
}

void ImagePair::setImageIndex(int index1, int index2){
    index_[0]=index1;
    index_[1]=index2;
}

/**
 * @brief   get image indexes
 * @return  image indexes
 */
std::array<int, 2> ImagePair::getImageIndex() const{
    return index_;
}

const std::vector<cv::DMatch>& ImagePair::getMatches() const {
	return matches_;
}

size_t ImagePair::getMatchNum() const {
	return matches_.size();
}

std::tuple<cv::Matx33d, cv::Matx31d> ImagePair::getExtrinsicParameter() const {
	return std::tuple<cv::Matx33d, cv::Matx31d>(rotation_mat_, translation_vec_	);
}

const std::vector<cv::Point3d>& ImagePair::getTriangulatedPoints() const {
	return triangulated_points_;
}

/**
 * @brief			run cross check
 * @param[in,out]	matches12	keypoint matchings from image1 to image2
 * @param[in,out]	matches21	keypoint matchings from image2 to image1
 */
void ImagePair::crossCheck(std::vector<std::vector<cv::DMatch>>& matches12, std::vector<std::vector<cv::DMatch>>& matches21) const {
	std::vector<std::vector<cv::DMatch>> good_matches12;
	std::vector<std::vector<cv::DMatch>> good_matches21;
	good_matches12.reserve(matches12.size());
	good_matches21.reserve(matches21.size());
	for (int i = 0; i < matches12.size(); i++) {
		const cv::DMatch kMatch12 = matches12[i][0];
		const cv::DMatch kMatch21 = matches21[kMatch12.trainIdx][0];
		if (kMatch21.trainIdx == kMatch12.queryIdx) {
			good_matches12.push_back(matches12[i]);
			good_matches21.push_back(matches21[kMatch12.trainIdx]);
		}
	}
	matches12 = std::move(good_matches12);
	matches21 = std::move(good_matches21);
}

/**
 * @brief		find good keypoints with distance of feature point space.
 *
 * Find good keypoints with distance of feature point space. 
 * The matchings are judged good if best matching distance / 2nd best matching distance < threshold.
 *  
 * @param[in]	kMatches12					keypoint matchings from image1 to image2
 * @param[in]	kMatches21					keypoint matchings from image2 to image1
 * @param[in]	distance_ratio_threshold	threshold of ratio of feature point space
 * @return		vector which is good matching. true means the good matching.	
 */
std::vector<bool> ImagePair::findGoodKeypointMatchingByDistanceRatio(const std::vector<std::vector<cv::DMatch>>& kMatches12, const std::vector<std::vector<cv::DMatch>>& kMatches21, double distance_ratio_threshold) const {
	std::vector<bool> is_good_matches(kMatches12.size(), true);

	for (int i = 0; i < kMatches12.size(); i++) {
		if (kMatches12[i].size() < 2) {
			is_good_matches.at(i) = false;
			continue;
		}
		if ((double)kMatches12[i][0].distance / (double)kMatches12[i][1].distance > distance_ratio_threshold) {
			is_good_matches.at(i) = false;
			continue;
		}
		if ((double)kMatches21[i][0].distance / (double)kMatches21[i][1].distance > distance_ratio_threshold) {
			is_good_matches.at(i) = false;
			continue;
		}
	}

	return std::move(is_good_matches);
}

/**
 * @brief		remove wrong matchings. the return value contains best matching of each matchings (2nd best matching is ignored).
 * @param[in]	kMatches		matchings
 * @param[in]	kIsGoodMatches	which are the good keypoints. true means good keypoints.
 * @return		good keypoint matchings
 */
std::vector<cv::DMatch> ImagePair::removeWrongKeypointMatching(const std::vector<std::vector<cv::DMatch>>& kMatches, const std::vector<bool>& kIsGoodMatches) const {
	std::vector<cv::DMatch> good_matches;
	for (int i = 0; i < kMatches.size(); i++) {
		if (kIsGoodMatches.at(i) == true) {
			good_matches.push_back(kMatches[i][0]);
		}
	}

	return std::move(good_matches);
}

cv::Mat1d ImagePair::findFundamentalMatrix(const cv::Size& kImageSize1, const std::vector<cv::KeyPoint>& kKeypoint1, const cv::Size& kImageSize2
	, const std::vector<cv::KeyPoint>& kKeypoint2, const std::vector<cv::DMatch>& kMatches, cv::Mat& is_good_matches) const {
	std::vector<cv::Point2d> good_keypoint1(kMatches.size());
	std::vector<cv::Point2d> good_keypoint2(kMatches.size());
	for (int i = 0; i < kMatches.size(); i++) {
		good_keypoint1.at(i) = kKeypoint1.at(kMatches[i].queryIdx).pt;
		good_keypoint2.at(i) = kKeypoint2.at(kMatches[i].trainIdx).pt;
	}

	const double kThreshold = std::max({ kImageSize1.width, kImageSize1.height, kImageSize2.width, kImageSize2.height })*0.006;
	cv::Mat1d fmat = cv::findFundamentalMat(good_keypoint1, good_keypoint2, CV_FM_RANSAC, kThreshold, 0.99, is_good_matches);

	return fmat;
}

/**
 * @brief		remove wrong matchings.
 * @param[in]	kMatches		matchings
 * @param[in]	kIsGoodMatches	which are the good keypoints. 1 means good keypoints.
 * @return		good keypoint matchings
 */
std::vector<cv::DMatch> ImagePair::removeWrongKeypointMatching(const std::vector<cv::DMatch>& kMatches, const cv::Mat & kIsGoodMatches) const {
	std::vector<cv::DMatch> good_matches;
	for (int i = 0; i < kMatches.size(); i++) {
		if (kIsGoodMatches.data[i] == 1) {
			good_matches.push_back(kMatches.at(i));
		}
	}

	return std::move(good_matches);
}
