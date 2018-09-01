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
	void tracking(int image_num, const std::vector<ImagePair>& kImagePairs);
	size_t getTrackingNum() const;
	bool isAmbiguousKeypoint(int image_index, int keypoint_index) const;
private:
	std::vector<std::unordered_map<int, int>> tracks_;	//! key=image index, value=keypoint index
	std::unordered_multimap<std::tuple<int, int>, std::tuple<int, int>> image_pair_map_;	//! key=<image index, keypoint index>, value=<image index, keypoint index>, <image index, keypoint index>...
	std::unordered_map<std::tuple<int, int>, bool> is_already_tracked_;	//! key=<image index, keypoint index>, value=true if target keypoint was already tracked (false mean ambiguous tracked).

	std::unordered_multimap<std::tuple<int, int>, std::tuple<int, int>> createImagePairMap(const std::vector<ImagePair>& kImagePairs) const;
	std::vector<std::unordered_map<int, int>> trackingAll(std::unordered_map<std::tuple<int, int>, bool>& is_alreadly_tracked
		, const std::unordered_multimap<std::tuple<int, int>, std::tuple<int, int>>& kImagePairMap) const;
	bool trackingOnce(std::unordered_map<int, int>& track, const std::tuple<int, int>& kKey, const std::unordered_multimap<std::tuple<int, int>, std::tuple<int, int>>& kImagePairMap) const;
	void updateTrackingState(std::unordered_map<std::tuple<int, int>, bool>& is_alreadly_tracked, const std::unordered_map<int, int>& track , bool state) const;
};

#endif