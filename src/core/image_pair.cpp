#include "image_pair.hpp"

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
	std::cout << "Keypoint size: " << kKeypoints1.size() << std::endl;
	std::cout << "Keypoint size: " << kKeypoints2.size() << std::endl;
	std::cout << "Before: " << matches12.size() << std::endl;
	std::cout << "Before: " << matches21.size() << std::endl;

	// cross check
	crossCheck(matches12, matches21);
	std::cout << "After2: " << matches12.size() << std::endl;
	std::cout << "After2: " << matches21.size() << std::endl;

	// remove wrong matching

}

/**
 * @brief       set image indexes
 * @param[in]   index1
 * @param[in]   index2
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

void ImagePair::crossCheck(std::vector<std::vector<cv::DMatch>>& matches12, std::vector<std::vector<cv::DMatch>>& matches21) const {
	std::vector<std::vector<cv::DMatch>> good_matches12;
	std::vector<std::vector<cv::DMatch>> good_matches21;
	good_matches12.reserve(matches12.size());
	good_matches21.reserve(matches21.size());
	for (int i = 0; i < matches12.size(); i++) {
		const cv::DMatch kForward = matches12[i][0];
		const cv::DMatch kBackward = matches21[kForward.trainIdx][0];
		if (kForward.trainIdx == kBackward.queryIdx) {
			good_matches12.push_back(matches12[i]);
			good_matches21.push_back(matches21[kForward.trainIdx]);
		}
	}
	matches12 = std::move(good_matches12);
	matches21 = std::move(good_matches21);
}

std::vector<cv::DMatch> ImagePair::findGoodKeypointMatching(const std::vector<std::vector<cv::DMatch>>& kMatches, double distance_ratio_threshold) const {
	std::vector<cv::DMatch> good_matches;
	good_matches.reserve(kMatches.size());
	for (const auto& match : kMatches) {
		if (match.size() < 2) {
			continue;
		}
		if (match[0].distance / match[1].distance < distance_ratio_threshold) {
			good_matches.push_back(match[0]);
		}
	}

	return std::move(good_matches);
}
