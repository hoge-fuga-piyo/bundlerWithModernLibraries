#ifndef SFM_HPP
#define SFM_HPP

#include <iostream>
#include <vector>
#include <array>
#include <opencv2/opencv.hpp>
#include "image.hpp"
#include "image_pair.hpp"
#include "fileUtil.hpp"
#include "tracking.hpp"

class SfM {
public:
	SfM();
	void loadImages(const std::string kDirPath);
	void detectKeypoints();
	void keypointMatching();
	void trackingKeypoint();
	void initialReconstruct();
private:
	const Image::DetectorType kDetectorType_;
	const int kMinimumInitialImagePairNum_;
	const double kHomographyThresholdRatio_;
	const double kDefaultFocalLength_;

	std::vector<Image> images_;
	std::vector<ImagePair> image_pair_;
	Tracking track_;

	int selectInitialImagePair(const std::vector<Image>& kImages, const std::vector<ImagePair>& kImagePair) const;
};

#endif