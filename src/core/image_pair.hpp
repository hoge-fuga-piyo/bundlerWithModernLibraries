#ifndef IMAGE_PAIR_HPP
#define IMAGE_PAIR_HPP

#include <iostream>
#include <vector>
#include <array>
#include <memory>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "image.hpp"

/**
 * @brief This class is about processing and information between 2 images.
 */
class ImagePair{
public:     // Methods
    ImagePair();
	void keypointMatching(const Image& kImage1, const Image& kImage2);
	void showMatches(const cv::Mat& kImg1, const std::vector<cv::KeyPoint>& kKeypoints1, const cv::Mat& kImg2, const std::vector<cv::KeyPoint>& kKeypoints2) const;
	double computeBaeslinePossibility(const Image& kImage1, const Image& kImage2, double homography_threshold) const;

public:     // Setter/Getter
    void setImageIndex(int index1, int index2);
    std::array<int, 2> getImageIndex() const;
	const std::vector<cv::DMatch>& getMatches() const;
	size_t getMatchNum() const;

private:    // Instance variables
	static const double kDistanceRatioThreshold_;
    std::array<int, 2> index_;		   //! index of two images
    std::vector<cv::DMatch> matches_;  //! information of keypoint matching

private:	// Methods
    std::vector<cv::DMatch> keypointMatching(const cv::Mat& kImage1, const std::vector<cv::KeyPoint>& kKeypoints1, const cv::Mat& kDescriptor1,
                        const cv::Mat& kImage2, const std::vector<cv::KeyPoint>& kKeypoints2, const cv::Mat& kDescriptor2);
	void crossCheck(std::vector<std::vector<cv::DMatch>> &matches12, std::vector<std::vector<cv::DMatch>> &matches21) const;
	std::vector<bool> findGoodKeypointMatchingByDistanceRatio(const std::vector<std::vector<cv::DMatch>>& kMatches12, const std::vector<std::vector<cv::DMatch>>& kMatches21, double distance_ratio_threshold) const;
	std::vector<cv::DMatch> removeWrongKeypointMatching(const std::vector<std::vector<cv::DMatch>>& kMatches, const std::vector<bool>& kIsGoodMatches) const;

	cv::Mat1d findFundamentalMatrix(const cv::Size& kImageSize1, const std::vector<cv::KeyPoint>& kKeypoint1
		, const cv::Size& kImageSize2, const std::vector<cv::KeyPoint>& kKeypoint2, const std::vector<cv::DMatch>& kMatches, cv::Mat& is_good_matches) const;
	std::vector<cv::DMatch> removeWrongKeypointMatching(const std::vector<cv::DMatch>& kMatches, const cv::Mat& kIsGoodMatches) const;
	double computeBaeslinePossibility(const std::vector<cv::KeyPoint>& kKeypoints1, const std::vector<cv::KeyPoint>& kKeypoints2, double homography_threshold) const;
};

#endif