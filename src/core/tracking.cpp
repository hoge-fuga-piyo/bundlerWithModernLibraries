#include "tracking.hpp"
#include <opencv2/viz.hpp>
#include <yaml-cpp/yaml.h>
#include "fileUtil.hpp"

Tracking::Tracking() {
	recovered_num_ = 0;
}

void Tracking::tracking(const std::vector<ImagePair>& kImagePairs) {
	image_pair_map_ = createImagePairMap(kImagePairs);
	tracks_ = trackingAll(is_already_tracked_, image_pair_map_);
	triangulated_points_.resize(tracks_.size());
	is_recovered_.resize(tracks_.size(), false);
	track_map_ = createTrackingMap(tracks_);
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
			if (trackingOnce(track, kValue.second, kImagePairMap) == false) {
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

std::unordered_map<std::tuple<int, int>, int>  Tracking::createTrackingMap(const std::vector<std::unordered_map<int, int>>& tracks) const {
	std::unordered_map<std::tuple<int, int>, int> track_map;
	for (int i = 0; i < (int)tracks.size(); i++) {
		for (auto itr = tracks[i].begin(); itr != tracks[i].end(); itr++) {
			track_map[std::tuple<int, int>(itr->first, itr->second)] = i;
		}
	}
	return std::move(track_map);
}

std::vector<cv::Vec3b> Tracking::extractPointColors(const std::vector<Image>& kImages) const {
	std::vector<cv::Vec3b> colors(tracks_.size());
	for (size_t i = 0; i < tracks_.size(); i++) {
		if(tracks_[i].size() == 0) {
			colors[i] = cv::Vec3b(0, 0, 0);
			continue;
		}
		cv::Vec3i bgr = cv::Vec3i(0, 0, 0);
		for (auto itr = tracks_[i].begin(); itr != tracks_[i].end(); itr++) {
			const cv::Vec3b kBgr = kImages[itr->first].getKeypointColor(itr->second);
			bgr += kBgr;
		}
		colors[i] = cv::Vec3b(static_cast<uchar>(bgr(0) / tracks_[i].size()), static_cast<uchar>(bgr(1) / tracks_[i].size()), static_cast<uchar>(bgr(2) / tracks_[i].size()));
	}

	return std::move(colors);
}

bool Tracking::isAmbiguousKeypoint(int image_index, int keypoint_index) const {
	bool state = is_already_tracked_.at(std::tuple<int, int>(image_index, keypoint_index));
	if (state == false) {
		return true;
	}
	return false;
}

void Tracking::setTriangulatedPoints(const ImagePair & kImagePair) {
	const std::array<int, 2> kImageIndex = kImagePair.getImageIndex();
	const std::vector<cv::DMatch>& kMatches = kImagePair.getMatches();
	const std::vector<cv::Point3d>& kTriangulatedPoints = kImagePair.getTriangulatedPoints();
	if (kMatches.size() != kTriangulatedPoints.size()) {
		std::cerr << "[ERROR] Different between the number of matches and number of triangulated points." << std::endl;
		return;
	}

	for (size_t i = 0; i < kMatches.size(); i++) {
		const std::tuple<int, int> kKey1 = std::tuple<int, int>(kImageIndex[0], kMatches[i].queryIdx);
		const std::tuple<int, int> kKey2 = std::tuple<int, int>(kImageIndex[1], kMatches[i].trainIdx);
		if (is_already_tracked_.count(kKey1) > 0 && is_already_tracked_[kKey1] == true
			&& is_already_tracked_.count(kKey2) > 0 && is_already_tracked_[kKey2] == true) {
			int track_index = track_map_.at(kKey1);
			triangulated_points_[track_index] = kTriangulatedPoints[i];
			if (is_recovered_[track_index] == false) {
				is_recovered_[track_index] = true;
				recovered_num_++;
			}
		}
	}
}

void Tracking::setTriangulatedPoint(int index, double x, double y, double z) {
	triangulated_points_.at(index) = cv::Point3d(x, y, z);
	if (!is_recovered_.at(index)) {
		recovered_num_++;
	}
	is_recovered_.at(index) = true;
}

void Tracking::saveTriangulatedPoints(const std::string & file_path, const std::vector<Image>& kImages) const {
	std::vector<cv::Vec3b> all_colors = extractPointColors(kImages);
	std::vector<cv::Vec3b> colors;
	std::vector<cv::Point3d> triangulated_points;
	for (size_t i = 0; i < is_recovered_.size(); i++) {
		if (is_recovered_.at(i) == true) {
			triangulated_points.push_back(triangulated_points_.at(i));
			colors.push_back(all_colors.at(i));
		}
	}
	cv::viz::writeCloud(file_path, triangulated_points, colors, cv::noArray(), false);
}

int Tracking::getTriangulatedPointNum() const {
	return recovered_num_;
}

bool Tracking::isRecoveredTriangulatedPoint(int index) const {
	return is_recovered_.at(index);
}

const cv::Point3d& Tracking::getTriangulatedPoint(int index) const {
	return triangulated_points_.at(index);
}

int Tracking::getTrackedKeypointIndex(int track_index, int image_index) const {
	if (tracks_[track_index].count(image_index) > 0) {
		return tracks_[track_index].at(image_index);
	}
	return -1;
}

std::vector<int> Tracking::countTriangulatedPointNum(int image_num) const {
	std::vector<int> recovered_point_num(image_num, 0);
	for (size_t i = 0; i < tracks_.size(); i++) {
		if (!is_recovered_[i]) {
			continue;
		}
		for (auto itr = tracks_[i].begin(); itr != tracks_[i].end(); itr++) {
			recovered_point_num[itr->first]++;
		}
	}

	return std::move(recovered_point_num);
}

void Tracking::extractImagePointAndWorlPointPairs(int image_index, const Image & kImage, std::vector<cv::Point2d>& image_points, std::vector<cv::Point3d>& world_points) const {
	image_points.clear();
	world_points.clear();
	const std::vector<cv::KeyPoint>& kKeypoints = kImage.getKeypoints();
	for (size_t i = 0; i < tracks_.size(); i++) {
		if (!is_recovered_[i]) {
			continue;
		}
		if (tracks_[i].find(image_index) == tracks_[i].end()) {
			continue;
		}

		int keypoint_index = tracks_[i].at(image_index);
		image_points.push_back(kKeypoints.at(keypoint_index).pt);
		world_points.push_back(triangulated_points_.at(i));
	}
}

void Tracking::removeTrack(int index) {
	tracks_.at(index).clear();
	if (is_recovered_.at(index)) {
		recovered_num_--;
	}
	is_recovered_.at(index) = false;
}

void Tracking::writeTrackingInfo(const std::string & kDirPath) const {
	YAML::Emitter out;
	out << YAML::BeginMap;
	
	// write tracking info
	out << YAML::Key << "tracks";
	out << YAML::Value;
	out << YAML::BeginSeq;
	for (const auto& track : tracks_) {
		std::vector<int> track_row;
		for (auto itr = track.begin(); itr != track.end(); ++itr) {
			track_row.push_back(itr->first);
			track_row.push_back(itr->second);
		}
		out << YAML::Flow << track_row;
	}
	out << YAML::EndSeq;

	// write track status
	out << YAML::Key << "track_status";
	out << YAML::Value;
	out << YAML::BeginSeq;
	for (auto itr = is_already_tracked_.begin(); itr != is_already_tracked_.end(); ++itr) {
		const std::vector<int> kStatus = { std::get<0>(itr->first), std::get<1>(itr->first), static_cast<int>(itr->second) };
		out << YAML::Flow << kStatus;
	}
	out << YAML::EndSeq;

	// write track map
	out << YAML::Key << "track_map";
	out << YAML::Value;
	out << YAML::BeginSeq;
	for (auto itr = track_map_.begin(); itr != track_map_.end(); ++itr) {
		const std::vector<int> kOneTrackMap = { std::get<0>(itr->first), std::get<1>(itr->first), itr->second };
		out << YAML::Flow << kOneTrackMap;
	}
	out << YAML::EndSeq;

	out << YAML::EndMap;

	const std::string kFilePath = FileUtil::addSlashToLast(kDirPath) + "trackInfo.yaml";
	std::ofstream ofs(kFilePath);
	ofs << out.c_str();
	ofs.close();
}

void Tracking::loadTrackingInfo(const std::string & kFilePath) {
	tracks_.clear();
	is_recovered_.clear();
	triangulated_points_.clear();
	YAML::Node log_file = YAML::LoadFile(kFilePath);

	// load tracking info
	tracks_.resize(log_file["tracks"].size());
	is_recovered_.resize(log_file["tracks"].size(), false);
	triangulated_points_.resize(log_file["tracks"].size());
	for (size_t i = 0; i < log_file["tracks"].size(); i++) {
		const std::vector<int> kOneTrackVec = log_file["tracks"][i].as<std::vector<int>>();
		std::unordered_map<int, int> one_track;
		for (size_t j = 0; j < kOneTrackVec.size() / 2; j++) {
			one_track[kOneTrackVec.at(j * 2 + 0)] = kOneTrackVec.at(j * 2 + 1);
		}
		tracks_[i] = one_track;
	}
	std::cout << "track num: " << tracks_.size() << std::endl;

	// load track status
	is_already_tracked_.clear();
	for (size_t i = 0; i < log_file["track_status"].size(); i++) {
		const std::vector<int> kStatus = log_file["track_status"][i].as<std::vector<int>>();
		if (kStatus.at(2) == 0) {
			is_already_tracked_[std::tuple<int, int>(kStatus.at(0), kStatus.at(1))] = false;
		} else {
			is_already_tracked_[std::tuple<int, int>(kStatus.at(0), kStatus.at(1))] = true;
		}
	}
	std::cout << "is_already: " << is_already_tracked_.size() << std::endl;

	// load track map
	track_map_.clear();
	for (size_t i = 0; i < log_file["track_map"].size(); i++) {
		const std::vector<int> kOneTrackMap = log_file["track_map"][i].as<std::vector<int>>();
		track_map_[std::tuple<int, int>(kOneTrackMap.at(0), kOneTrackMap.at(1))] = kOneTrackMap.at(2);
	}
	std::cout << "track_map: " << track_map_.size() << std::endl;
}
