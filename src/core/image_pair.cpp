#include "image_pair.hpp"

const double ImagePair::kDistanceRatioThreshold_ = 0.6;

ImagePair::ImagePair(){
}

void ImagePair::keypointMatching(const std::vector<cv::KeyPoint>& kKeypoints1
	, const cv::Mat& kDescriptor1
	, const std::vector<cv::KeyPoint>& kKeypoints2
	, const cv::Mat& kDescriptor2){

	// keypoint matching
	cv::FlannBasedMatcher matcher;
	std::vector<std::vector<cv::DMatch>> matches12, matches21;
	matcher.knnMatch(kDescriptor1, kDescriptor2, matches12, 2);
	matcher.knnMatch(kDescriptor2, kDescriptor1, matches21, 2);

	// cross check
	crossCheck(matches12, matches21);
	//std::vector<bool> tmp(matches12.size(), true);
	//matches_ = removeWrongKeypointMatching(matches12, tmp);

	// remove wrong matching
	std::vector<bool> is_good_matches = findGoodKeypointMatching(matches12, matches21, kDistanceRatioThreshold_);
	matches_ = removeWrongKeypointMatching(matches12, is_good_matches);
}

void ImagePair::showMatches(const cv::Mat & kImg1, const std::vector<cv::KeyPoint>& kKeypoints1, const cv::Mat & kImg2, const std::vector<cv::KeyPoint>& kKeypoints2) const {
	cv::Mat dst_img;
	cv::drawMatches(kImg1, kKeypoints1, kImg2, kKeypoints2, matches_, dst_img);
	cv::imshow("", dst_img);
	cv::waitKey(0);
}

/**
 * @brief       set image indexes
 * @param[in]   index1	image index of image1
 * @param[in]   index2	image index of image2
 */
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
std::vector<bool> ImagePair::findGoodKeypointMatching(const std::vector<std::vector<cv::DMatch>>& kMatches12, const std::vector<std::vector<cv::DMatch>>& kMatches21, double distance_ratio_threshold) const {
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
 * @param[in]	kIsGoodMatches	which are the good keypoints
 * @return		good keypoint matchings
 */
std::vector<cv::DMatch> ImagePair::removeWrongKeypointMatching(const std::vector<std::vector<cv::DMatch>>& kMatches, const std::vector<bool>& kIsGoodMatches) const {
	std::vector<cv::DMatch> good_matches;
	std::cout << kMatches.size() << std::endl;
	for (int i = 0; i < kMatches.size(); i++) {
		if (kIsGoodMatches.at(i) == true) {
			good_matches.push_back(kMatches[i][0]);
		}
	}

	return std::move(good_matches);
}
