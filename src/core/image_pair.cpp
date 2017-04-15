#include "image_pair.hpp"

ImagePair::ImagePair(){

}

void ImagePair::keypointMatching(const std::vector<cv::KeyPoint>& kKeypoints1, const cv::Mat& descriptor1,
                        const std::vector<cv::KeyPoint>& kKeypoint2, const cv::Mat& descriptor2){

}

/**
 * @brief       set image indexes
 * @param[in]   index1
 * @param[in]   index2
 */
void ImagePair::setImageIndex(int index1, int index2) const{
    index[0]=index1;
    index[1]=index2;
}

/**
 * @brief   get image indexes
 * @return  image indexes
 */
std::array<int, 2> ImagePair::getImageIndex() const{
    return index;
}