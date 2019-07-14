#ifndef TRACKING_HPP
#define TRACKING_HPP

#include <iostream>
#include <vector>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include "image_pair.hpp"
#include "image.hpp"
#include "hash_key.hpp"

class Tracking {
public:
	Tracking();
	void tracking(int image_num, const std::vector<ImagePair>& kImagePairs);
	size_t getTrackingNum() const;
	bool isAmbiguousKeypoint(int image_index, int keypoint_index) const;
	void setTriangulatedPoints(const ImagePair& kImagePair);
	void setTriangulatedPoint(int index, double x, double y, double z);
	void saveTriangulatedPoints(const std::string& file_path, const std::vector<Image>& kImages) const;
	int getTriangulatedPointNum() const;
	bool isRecoveredTriangulatedPoint(int index) const;
	const cv::Point3d& getTriangulatedPoint(int index) const;
	int getTrackedKeypointIndex(int track_index, int image_index) const;
	std::vector<int> countTriangulatedPointNum(int image_num) const;
	void extractImagePointAndWorlPointPairs(int image_index, const Image& kImage, std::vector<cv::Point2d>& image_points, std::vector<cv::Point3d>& world_points) const;
	void removeTrack(int index);

	void writeTrackingInfo(const std::string& kDirPath) const;
	void loadTrackingInfo(const std::string& kFilePath);

private:
	std::vector<std::unordered_map<int, int>> tracks_;	//! key=image index, value=keypoint index
	std::unordered_multimap<std::tuple<int, int>, std::tuple<int, int>> image_pair_map_;	//! key=<image index, keypoint index>, value=<image index, keypoint index>, <image index, keypoint index>...
	std::unordered_map<std::tuple<int, int>, bool> is_already_tracked_;	//! key=<image index, keypoint index>, value=true if target keypoint was already tracked (false mean ambiguous tracked).
	std::vector<cv::Point3d> triangulated_points_;
	std::unordered_map<std::tuple<int, int>, int> track_map_;
	std::vector<bool> is_recovered_;
	int recovered_num_;

	std::unordered_multimap<std::tuple<int, int>, std::tuple<int, int>> createImagePairMap(const std::vector<ImagePair>& kImagePairs) const;
	std::vector<std::unordered_map<int, int>> trackingAll(std::unordered_map<std::tuple<int, int>, bool>& is_alreadly_tracked
		, const std::unordered_multimap<std::tuple<int, int>, std::tuple<int, int>>& kImagePairMap) const;
	bool trackingOnce(std::unordered_map<int, int>& track, const std::tuple<int, int>& kKey, const std::unordered_multimap<std::tuple<int, int>, std::tuple<int, int>>& kImagePairMap) const;
	void updateTrackingState(std::unordered_map<std::tuple<int, int>, bool>& is_alreadly_tracked, const std::unordered_map<int, int>& track , bool state) const;
	std::unordered_map<std::tuple<int, int>, int> createTrackingMap(const std::vector<std::unordered_map<int, int>>& tracks) const;
	std::vector<cv::Vec3b> extractPointColors(const std::vector<Image>& kImages) const;
};

#endif