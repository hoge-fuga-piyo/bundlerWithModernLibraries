#ifndef TRACKING_HPP
#define TRACKING_HPP

#include <iostream>
#include <vector>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include "image_pair.hpp"
#include "image.hpp"

class Tracking {
public:
	void tracking(int image_num, const std::vector<ImagePair>& kImagePairs);
	size_t getTrackingNum() const;
private:
	enum class KeypointState {
		TRACKED,
		AMBIGUOUS
	};

	std::vector<std::unordered_map<int, int>> track_;	//! tracking result. key is image index, value is keypoint index
	std::vector<std::unordered_map<int, KeypointState>> keypoint_map_;//! 

	std::vector<std::unordered_map<int, int>> trackingAll(const std::vector<ImagePair>& kImagePairs, std::vector<std::unordered_map<int, KeypointState>>& keypoint_map) const;
	void trackingOneImagePair(int image_index1, int image_index2, const std::vector<cv::DMatch>& kMatches, std::vector<std::unordered_map<int, int>>& all_track
		, std::vector<std::unordered_map<int, KeypointState>>& keypoint_map) const;
	std::unordered_map<int, int> trackingOneMatch(int image_index1, int image_index2, const cv::DMatch& kMatch, std::vector<std::unordered_map<int, int>>& all_track
		, std::vector<std::unordered_map<int, KeypointState>>& keypoint_map, std::unordered_map<int, bool>& ignore_track_index) const;
	bool isAlreadyTracked(int image_index1, int image_index2, const cv::DMatch& kMatch, const std::unordered_map<int, int>& kTrack) const;
	bool isAmbiguousTracked(int image_index1, int image_index2, const cv::DMatch& kMatch, const std::unordered_map<int, int>& kTrack) const;
	void markAmbiguousTracked(const std::unordered_map<int, int>& kTrack, std::vector<std::unordered_map<int, KeypointState>>& keypoint_map) const;
};

#endif