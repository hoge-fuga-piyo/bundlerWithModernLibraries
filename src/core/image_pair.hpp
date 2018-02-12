#include <iostream>
#include <vector>
#include <array>
#include <opencv2/opencv.hpp>

/**
 * @brief This class is about processing and information between 2 images.
 */
class ImagePair{
public:     // Methods
    ImagePair();
    void keypointMatching(const std::vector<cv::KeyPoint>& kKeypoints1, const cv::Mat& kDescriptor1,
                        const std::vector<cv::KeyPoint>& kKeypoint2, const cv::Mat& kDescriptor2);

public:     // Setter/Getter
    void setImageIndex(int index1, int index2);
    std::array<int, 2> getImageIndex() const;

private:    // Instance variables
    std::array<int, 2> index_;       //! index of two images
    std::vector<cv::DMatch> match_;  //! information of keypoint matching

private:	// Methods
	void crossCheck(std::vector<std::vector<cv::DMatch>> &matches12, std::vector<std::vector<cv::DMatch>> &matches21) const;
	std::vector<cv::DMatch> findGoodKeypointMatching(const std::vector<std::vector<cv::DMatch>>& kMatchs, double distance_ratio) const;
};