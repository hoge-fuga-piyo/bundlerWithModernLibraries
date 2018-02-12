#include "image_pair.hpp"

ImagePair::ImagePair(){

}

void ImagePair::keypointMatching(const std::vector<cv::KeyPoint>& kKeypoints1
	, const cv::Mat& kDescriptor1
	, const std::vector<cv::KeyPoint>& kKeypoint2
	, const cv::Mat& kDescriptor2){

	// keypoint matching
	cv::FlannBasedMatcher matcher;
	std::vector<std::vector<cv::DMatch>> matches12, matches21;
	matcher.knnMatch(kDescriptor1, kDescriptor2, matches12, 2);
	matcher.knnMatch(kDescriptor1, kDescriptor2, matches21, 2);
	std::cout << "Before1: " << matches12.size() << std::endl;
	std::cout << "Before2: " << matches21.size() << std::endl;

	// cross check
	crossCheck(matches12, matches21);
	std::cout << "After1: " << matches12.size() << std::endl;
	std::cout << "After2: " << matches21.size() << std::endl;
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
	for (const auto& match12 : matches12) {
		for (const auto& match21 : matches21) {
			if (match12.at(0).trainIdx == match21.at(0).queryIdx) {
				good_matches12.push_back(match12);
				good_matches21.push_back(match21);
			}
		}
	}
	matches12 = std::move(good_matches12);
	matches21 = std::move(good_matches21);
}

std::vector<cv::DMatch> ImagePair::findGoodKeypointMatching(const std::vector<std::vector<cv::DMatch>>& kMatchs, double distance_ratio) const {

	return std::vector<cv::DMatch>();
}
