#include <iostream>
#include <vector>
#include <array>
#include <opencv2/opencv.hpp>

/**
 * @brief This class is about processing and information between 2 images.
 */
class ImagePair{
public:     //Methods
    ImagePair();
    void keypointMatching(const std::vector<cv::KeyPoint>& kKeypoints1, const cv::Mat& descriptor1,
                        const std::vector<cv::KeyPoint>& kKeypoint2, const cv::Mat& descriptor2);

public:     //Setter/Getter
    void setImageIndex(int index1, int index2) const;
    std::array<int, 2> getImageIndex() const;

private:    //Instance variables
    std::array<int, 2> index;       //! index of two images
    std::vector<cv::DMatch> match;  //! information of keypoint matching
};