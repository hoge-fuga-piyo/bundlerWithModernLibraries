#include "tracking.hpp"
#include "hash_key.hpp"

void Tracking::tracking(int image_num, const std::vector<ImagePair>& kImagePairs) {
	keypoint_map_.resize(image_num);
	track_ = trackingAll(kImagePairs, keypoint_map_);
}

size_t Tracking::getTrackingNum() const {
	return track_.size();
}

std::vector<std::unordered_map<int, int>> Tracking::trackingAll(const std::vector<ImagePair>& kImagePairs, std::vector<std::unordered_map<int, KeypointState>>& keypoint_map) const {
	std::vector<std::unordered_map<int, int>> all_track;	// key is image index, value is keypoint index
	int i = 0;
	for (const auto& kPair : kImagePairs) {
		std::cout << i << "-th pair tracking" << std::endl;
		i++;
		const std::array<int, 2>& kImageIndex = kPair.getImageIndex();
		const std::vector<cv::DMatch>& kMatches = kPair.getMatches();
		trackingOneImagePair(kImageIndex[0], kImageIndex[1], kMatches, all_track, keypoint_map);
	}

	return std::move(all_track);
}

void Tracking::trackingOneImagePair(int image_index1, int image_index2, const std::vector<cv::DMatch>& kMatches, std::vector<std::unordered_map<int, int>>& all_track
	, std::vector<std::unordered_map<int, KeypointState>>& keypoint_map) const {
	std::unordered_map<int, bool> ignore_track_index;
	for (const auto& kMatch : kMatches) {
		std::unordered_map<int, int> new_track = trackingOneMatch(image_index1, image_index2, kMatch, all_track, keypoint_map, ignore_track_index);
		if (!new_track.empty()) {
			all_track.push_back(new_track);
			ignore_track_index[all_track.size() - 1] = true;
		}
	}
}

std::unordered_map<int, int> Tracking::trackingOneMatch(int image_index1, int image_index2, const cv::DMatch & kMatch, std::vector<std::unordered_map<int, int>>& all_track
	, std::vector<std::unordered_map<int, KeypointState>>& keypoint_map, std::unordered_map<int, bool>& ignore_track_index) const {
	for (auto& track : all_track) {
		if (isAlreadyTracked(image_index1, image_index2, kMatch, track)==true) {
			return std::unordered_map<int, int>();
		}

		if (isAmbiguousTracked(image_index1, image_index2, kMatch, track) == true) {
			markAmbiguousTracked(track, keypoint_map);
			keypoint_map.at(image_index1)[kMatch.queryIdx] = KeypointState::AMBIGUOUS;
			keypoint_map.at(image_index2)[kMatch.trainIdx] = KeypointState::AMBIGUOUS;
			return std::unordered_map<int, int>();
		}

		if (track.count(image_index1) > 0 && track.at(image_index1)==kMatch.queryIdx) {
			track[image_index2] = kMatch.trainIdx;
			keypoint_map.at(image_index2)[kMatch.trainIdx] = KeypointState::TRACKED;
			return std::unordered_map<int, int>();
		} else if (track.count(image_index2) > 0 && track.at(image_index2)==kMatch.trainIdx) {
			track[image_index1] = kMatch.queryIdx;
			keypoint_map.at(image_index1)[kMatch.queryIdx] = KeypointState::TRACKED;
			return std::unordered_map<int, int>();
		}
	}

	std::unordered_map<int, int> new_track;
	new_track[image_index1] = kMatch.queryIdx;
	new_track[image_index2] = kMatch.trainIdx;
	keypoint_map.at(image_index1)[kMatch.queryIdx] = KeypointState::TRACKED;
	keypoint_map.at(image_index2)[kMatch.trainIdx] = KeypointState::TRACKED;
	return std::move(new_track);
}

bool Tracking::isAlreadyTracked(int image_index1, int image_index2, const cv::DMatch & kMatch, const std::unordered_map<int, int>& kTrack) const {
	if (!(kTrack.count(image_index1) > 0 && kTrack.at(image_index1) == kMatch.queryIdx)) {
		return false;
	}
	if (!(kTrack.count(image_index2) > 0 && kTrack.at(image_index2) == kMatch.trainIdx)) {
		return false;
	}

	return true;
}

bool Tracking::isAmbiguousTracked(int image_index1, int image_index2, const cv::DMatch & kMatch, const std::unordered_map<int, int>& kTrack) const {
	if (kTrack.count(image_index1) > 0 && kTrack.at(image_index1) == kMatch.queryIdx) {
		if (kTrack.count(image_index2) > 0 && kTrack.at(image_index2) != kMatch.trainIdx) {
			return true;
		}
	}

	if (kTrack.count(image_index2) > 0 && kTrack.at(image_index2) == kMatch.trainIdx) {
		if (kTrack.count(image_index1) > 0 && kTrack.at(image_index1) != kMatch.queryIdx) {
			return true;
		}
	}

	return false;
}

void Tracking::markAmbiguousTracked(const std::unordered_map<int, int>& kTrack, std::vector<std::unordered_map<int, KeypointState>>& keypoint_map) const {
	for (const auto& kTrackElement : kTrack) {
		keypoint_map.at(kTrackElement.first)[kTrackElement.second] = KeypointState::AMBIGUOUS;
	}
}
