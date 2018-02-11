#include "image_pair.hpp"

ImagePair::ImagePair(){

}

void ImagePair::keypointMatching(const std::vector<cv::KeyPoint>& kKeypoints1
	, const cv::Mat& kDescriptor1
	, const std::vector<cv::KeyPoint>& kKeypoint2
	, const cv::Mat& kDescriptor2){

	cv::FlannBasedMatcher matcher;
	std::vector<std::vector<cv::DMatch>> match;
	matcher.knnMatch(kDescriptor1, kDescriptor2, match, 2);
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