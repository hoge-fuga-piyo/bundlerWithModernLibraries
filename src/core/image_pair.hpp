#include <iostream>
#include <vector>
#include <array>
#include <memory>
#include <opencv2/opencv.hpp>

/**
 * @brief This class is about processing and information between 2 images.
 */
class ImagePair{
public:     // Methods
    ImagePair();
    void keypointMatching(const std::vector<cv::KeyPoint>& kKeypoints1, const cv::Mat& kDescriptor1,
                        const std::vector<cv::KeyPoint>& kKeypoints2, const cv::Mat& kDescriptor2);
	void showMatches(const cv::Mat& kImg1, const std::vector<cv::KeyPoint>& kKeypoints1, const cv::Mat& kImg2, const std::vector<cv::KeyPoint>& kKeypoints2) const;

public:     // Setter/Getter
    void setImageIndex(int index1, int index2);
    std::array<int, 2> getImageIndex() const;

private:    // Instance variables
    std::array<int, 2> index_;       //! index of two images
    std::vector<cv::DMatch> matches_;  //! information of keypoint matching

private:	// Methods
	void crossCheck(std::vector<std::vector<cv::DMatch>> &matches12, std::vector<std::vector<cv::DMatch>> &matches21) const;
	std::vector<bool> findGoodKeypointMatching(const std::vector<std::vector<cv::DMatch>>& kMatches12, const std::vector<std::vector<cv::DMatch>>& kMatches21, double distance_ratio_threshold) const;
	std::vector<cv::DMatch> removeWrongKeypointMatching(const std::vector<std::vector<cv::DMatch>>& kMatches, const std::vector<bool>& kIsGoodMatches) const;
};