#ifndef SFM_HPP
#define SFM_HPP

#include <iostream>
#include <vector>
#include <array>
#include <opencv2/opencv.hpp>
#include "image.hpp"
#include "image_pair.hpp"
#include "fileUtil.hpp"

class SfM {
public:
	void loadImages(const std::string kDirPath);
	void detectKeypoints();
	void keypointMatching();
private:
	const Image::DetectorType kDetectorType = Image::DetectorType::SIFT;

	std::vector<Image> images_;
	std::vector<ImagePair> image_pair_;
};

#endif