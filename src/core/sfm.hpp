#ifndef SFM_HPP
#define SFM_HPP

#include <iostream>
#include <vector>
#include <array>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <glog/logging.h>
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
	bool nextReconstruct();

	void savePointCloud(const std::string& file_path) const;
private:
	const Image::DetectorType kDetectorType_;
	const int kMinimumInitialImagePairNum_;
	const double kHomographyThresholdRatio_;
	const double kDefaultFocalLength_;
	const double kInfinityPointAngleDegree_;
	const int kPointCorrespondenceThresholdForCameraPoseRecover_;

	std::vector<Image> images_;
	std::vector<ImagePair> image_pair_;
	Tracking track_;

	int selectInitialImagePair(const std::vector<Image>& kImages, const std::vector<ImagePair>& kImagePair) const;
	void optimization(Tracking& track, std::vector<Image>& images) const;
	int selectNextReconstructImage(const Tracking& kTrack, const std::vector<Image>& kImages) const;
	std::vector<int> selectNextReconstructImages(const Tracking& kTrack, const std::vector<Image>& kImages) const;
	void computeNewObservedWorldPoints(int image_index, const std::vector<Image>& kImages, Tracking& track) const;
	bool isInfinityPoint(double degree_threshold, const cv::Point3d& kTriangulatedPoint, const std::vector<cv::Matx33d>& kRotationMatrix, const std::vector<cv::Matx31d>& kTranslationVector) const;
	bool removeHighReprojectionErrorTracks(Tracking& track, const std::vector<Image>& kImages) const;
};

#endif