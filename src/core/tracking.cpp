#include "tracking.hpp"

void Tracking::tracking(int image_num, const std::vector<ImagePair>& kImagePairs) {
	image_pair_map_ = createImagePairMap(kImagePairs);
	tracks_ = trackingAll(is_already_tracked_, image_pair_map_);
}

size_t Tracking::getTrackingNum() const {
	return tracks_.size();
}

std::unordered_multimap<std::tuple<int, int>, std::tuple<int, int>> Tracking::createImagePairMap(const std::vector<ImagePair>& kImagePairs) const {
	using KEY = std::tuple<int, int>;
	using VALUE = std::tuple<int, int>;

	std::unordered_multimap<KEY, VALUE> image_pair_map;
	for (const auto& kImagePair : kImagePairs) {
		const std::array<int, 2> kImageIndex = kImagePair.getImageIndex();
		const std::vector<cv::DMatch> kMatches = kImagePair.getMatches();
		for (const auto& kMatch : kMatches) {
			image_pair_map.insert(std::pair<KEY, VALUE>(KEY(kImageIndex[0], kMatch.queryIdx), VALUE(kImageIndex[1], kMatch.trainIdx)));
			image_pair_map.insert(std::pair<KEY, VALUE>(KEY(kImageIndex[1], kMatch.trainIdx), VALUE(kImageIndex[0], kMatch.queryIdx)));
		}
	}

	return std::move(image_pair_map);
}

std::vector<std::unordered_map<int, int>> Tracking::trackingAll(std::unordered_map<std::tuple<int, int>, bool>& is_alreadly_tracked
	, const std::unordered_multimap<std::tuple<int, int>, std::tuple<int, int>>& kImagePairMap) const {

	std::vector<std::unordered_map<int, int>> tracks;
	for (const auto& kPairMap : kImagePairMap) {
		const std::tuple<int, int> kKey = kPairMap.first;
		if (is_alreadly_tracked.count(kKey) > 0) {
			continue;
		}

		std::unordered_map<int, int> track;
		bool is_valid = trackingOnce(track, kKey, kImagePairMap);
		if (is_valid) {
			tracks.push_back(track);
		}
		updateTrackingState(is_alreadly_tracked, track, is_valid);
	}

	return std::move(tracks);
}

bool Tracking::trackingOnce(std::unordered_map<int, int>& track, const std::tuple<int, int>& kKey, const std::unordered_multimap<std::tuple<int, int>, std::tuple<int, int>>& kImagePairMap) const {
	if (track.count(std::get<0>(kKey)) > 0 && track.at(std::get<0>(kKey)) != std::get<1>(kKey)) {
		std::cerr << "[ERROR]: Unexpected Error in " << __LINE__ << "of" << __FILE__ << std::endl;
	}

	track[std::get<0>(kKey)] = std::get<1>(kKey);
	const std::pair<std::unordered_multimap<std::tuple<int, int>, std::tuple<int, int>>::const_iterator, std::unordered_multimap<std::tuple<int, int>, std::tuple<int, int>>::const_iterator> kValues = kImagePairMap.equal_range(kKey);

	for (auto itr = kValues.first; itr != kValues.second; itr++) {
		const std::pair<std::tuple<int, int>, std::tuple<int, int>> kValue = *itr;
		const int kImageIndex = std::get<0>(kValue.second);
		const int kKeypointIndex = std::get<1>(kValue.second);

		if (track.count(kImageIndex) == 0) {
		 	track[kImageIndex] = kKeypointIndex;
			if (trackingOnce(track, kValue.second, kImagePairMap)==false) {
				return false;
			}
		}
		else if (track.at(kImageIndex) != kKeypointIndex) {	// ambiguous
			return false;
		}
	}

	return true;
}

void Tracking::updateTrackingState(std::unordered_map<std::tuple<int, int>, bool>& is_alreadly_tracked, const std::unordered_map<int, int>& track, bool state) const {
	for (const auto& element : track) {
		const std::tuple<int, int> kKey = std::tuple<int, int>(element.first, element.second);
		if (state == true) {
			if (is_alreadly_tracked.count(kKey)>0 && is_alreadly_tracked.at(kKey) == false) {
				std::cerr << "[ERROR]: Unexpected Error in " << __LINE__ << "of" << __FILE__ << std::endl;
			}
		}
		is_alreadly_tracked[kKey] = state;
	}
}

bool Tracking::isAmbiguousKeypoint(int image_index, int keypoint_index) const {
	bool state = is_already_tracked_.at(std::tuple<int, int>(image_index, keypoint_index));
	if (state == false) {
		return true;
	}
	return false;
}