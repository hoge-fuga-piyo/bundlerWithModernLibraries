#include "sfm.hpp"
#include "bundleAdjustment.hpp"
#include "cvUtil.hpp"
#include "mathUtil.hpp"

SfM::SfM() : kDetectorType_(Image::DetectorType::SIFT)
, kMinimumInitialImagePairNum_(100)
, kHomographyThresholdRatio_(0.4)
, kDefaultFocalLength_(532.0)
, kInfinityPointAngleDegree_(2.0)
, kPointCorrespondenceThresholdForCameraPoseRecover_(20) {
}

void SfM::loadImagesAndDetectKeypoints(const std::string kDirPath) {
	const std::vector<std::experimental::filesystem::path> kFilePaths = FileUtil::readFiles(kDirPath);
	for (const auto& kPath : kFilePaths) {
		const std::string kExtension = kPath.extension().string();
		if (kExtension != ".jpg" && kExtension != ".JPG" && kExtension != ".png" && kExtension != ".PNG") {
			continue;
		}
		std::cout << "Load " << kPath.string() << "..." << std::endl;
		Image image;
		//image.loadImage(kPath.string());
		image.loadAndDetectKeypoints(kPath.string(), kDetectorType_);
		image.setFocalLength(kDefaultFocalLength_);
		image.setFileName(kPath.filename().string());
		images_.push_back(image);
		std::cout << " done." << std::endl;
	}
}

//void SfM::detectKeypoints() {
//	int kImageNum = static_cast<int>(images_.size());
//#pragma omp parallel for
//	for (int i = 0; i < kImageNum; i++) {
//		std::cout << "Detecting keypoints of" << i << "-th image." << std::endl;
//		images_[i].detectKeyPoints(kDetectorType_);
//	}
//}

void SfM::keypointMatching() {
	int kImageNum = static_cast<int>(images_.size());
	const unsigned long long kCombinationNum = MathUtil::combination(static_cast<unsigned int>(kImageNum), 2);
	std::cout << "Image num: " << kImageNum << std::endl;
	std::cout << "Combination num: "<<kCombinationNum << std::endl;
	image_pair_.resize(kCombinationNum);

	for (int i = 0; i < kImageNum; i++) {
		int index = 0;
		for (int j = 1; j <= i; j++) {
			index += kImageNum - j;
		}
#pragma omp parallel for
		for (int j = i + 1; j < kImageNum; j++) {
			std::cout << "Matching " << i << "-th image and " << j << "-th image" << std::endl;
			std::cout << "Thread ID: " << omp_get_thread_num() << std::endl;
			ImagePair image_pair;
			image_pair.setImageIndex(i, j);
			image_pair.keypointMatching(images_[i], images_[j]);
			//image_pair.showMatches(images_[i].getImage(), images_[i].getKeypoints(), images_[j].getImage(), images_[j].getKeypoints());
			//image_pair_.push_back(image_pair);
			image_pair_.at(index + (j - i - 1)) = image_pair;
		}
	}
	std::cout << image_pair_.size() << " image pairs are found." << std::endl;
}

void SfM::trackingKeypoint() {
	track_.tracking(static_cast<int>(images_.size()), image_pair_);
	std::cout << "Tracking num: " << track_.getTrackingNum() << std::endl;
}

void SfM::initialReconstruct() {
	int initial_pair_index = selectInitialImagePair(images_, image_pair_);
	const std::array<int, 2> initial_image_index = image_pair_[initial_pair_index].getImageIndex();
	std::cout << "Initial image pair are " << initial_image_index.at(0) << " and " << initial_image_index.at(1) << std::endl;

	image_pair_[initial_pair_index].recoverStructureAndMotion(images_[initial_image_index.at(0)], images_[initial_image_index.at(1)]);
	const cv::Matx33d& kRotationMat1 = cv::Matx33d::eye();
	const cv::Matx31d& kTranslationVec1 = cv::Matx31d::zeros();
	const cv::Matx33d& kRotationMat2 = image_pair_[initial_pair_index].getRotationMat();
	const cv::Matx31d& kTranslationVec2 = image_pair_[initial_pair_index].getTranslation();
	images_[initial_image_index.at(0)].setExtrinsicParameter(kRotationMat1, kTranslationVec1);
	images_[initial_image_index.at(1)].setExtrinsicParameter(kRotationMat2, kTranslationVec2);
	track_.setTriangulatedPoints(image_pair_[initial_pair_index]);

	std::cout << "Recovered num: " << track_.getTriangulatedPointNum() << std::endl;

	std::cout << "Optimization..." << std::endl;

	while (true) {
		optimization(track_, images_);
		bool doRemoving = removeHighReprojectionErrorTracks(track_, images_);
		if (!doRemoving) {
			break;
		}
	}
}

bool SfM::nextReconstruct() {
	std::vector<int> next_image_indexes = selectNextReconstructImages(track_, images_);
	if (next_image_indexes.size() == 0) {
		return false;
	}

	for (const auto index : next_image_indexes) {
		std::cout << "Next index: " << index << std::endl;

		std::vector<cv::Point2d> image_points;
		std::vector<cv::Point3d> world_points;
		track_.extractImagePointAndWorlPointPairs(index, images_[index], image_points, world_points);
		const cv::Size2i kImageSize = images_[index].getImageSize();
		const double threshold = std::max(kImageSize.width, kImageSize.height) * 0.004;
		std::cout << "threshold: " << threshold << std::endl;
		const cv::Matx34d kCameraParam = CvUtil::computeCameraParameterUsingRansac(image_points, world_points, threshold, 0.5, 0.9999);

		cv::Matx33d intrinsic_param;
		cv::Matx33d rotation_mat;
		cv::Matx31d translation_vec;
		CvUtil::decomposeProjectionMatrix(kCameraParam, intrinsic_param, rotation_mat, translation_vec);

		images_[index].setExtrinsicParameter(rotation_mat, translation_vec);
		images_[index].setFocalLength((intrinsic_param(0, 0) + intrinsic_param(1, 1)) / 2.0);
		//images_[index].setPrincipalPoint(intrinsic_param(0, 2), intrinsic_param(1, 2));
	}

	BundleAdjustment bundle_adjustment;
	bundle_adjustment.runBundleAdjustment(images_, track_, next_image_indexes);

	for (const auto index : next_image_indexes) {
		computeNewObservedWorldPoints(index, images_, track_);
	}

	while (true) {
		bundle_adjustment.runBundleAdjustment(images_, track_);
		bool doRemoving = removeHighReprojectionErrorTracks(track_, images_);
		if (!doRemoving) {
			break;
		}
	}

	return true;
}

int SfM::selectNextReconstructImage(const Tracking & kTrack, const std::vector<Image>& kImages) const {
	std::vector<int> recoverd_point_num = kTrack.countTriangulatedPointNum(static_cast<int>(kImages.size()));
	int max_num = 0;
	int next_image_index = -1;
	for (size_t i = 0; i < kImages.size(); i++) {
		if (kImages[i].isRecoveredExtrinsicParameter()) {
			continue;
		}
		std::cout << recoverd_point_num[i] << std::endl;
		if (max_num < recoverd_point_num[i]) {
			max_num = recoverd_point_num[i];
			next_image_index = static_cast<int>(i);
		}
	}

	if (max_num < kPointCorrespondenceThresholdForCameraPoseRecover_) {
		return -1;
	}

	return next_image_index;
}

std::vector<int> SfM::selectNextReconstructImages(const Tracking & kTrack, const std::vector<Image>& kImages) const {
	std::vector<int> recovered_point_num = kTrack.countTriangulatedPointNum(static_cast<int>(kImages.size()));
	int max_num = 0;
	int max_image_index = -1;
	for (size_t i = 0; i < kImages.size(); i++) {
		if (kImages[i].isRecoveredExtrinsicParameter()) {
			continue;
		}
		if (max_num < recovered_point_num[i]) {
			max_num = recovered_point_num[i];
			max_image_index = static_cast<int>(i);
		}
	}

	if (max_num < kPointCorrespondenceThresholdForCameraPoseRecover_) {
		return std::vector<int>();
	}
	
	std::vector<int> next_image_indexes;
	for (size_t i = 0; i < kImages.size(); i++) {
		if (kImages[i].isRecoveredExtrinsicParameter()) {
			continue;
		}
		if (recovered_point_num[i] > 0.75 * recovered_point_num[max_image_index]) {
			next_image_indexes.push_back(static_cast<int>(i));
		}
		std::cout << i << " : " << recovered_point_num[i] << std::endl;
	}

	return std::move(next_image_indexes);
}

void SfM::computeNewObservedWorldPoints(int image_index, const std::vector<Image>& kImages, Tracking& track) const {
	const std::vector<cv::KeyPoint>& kTargetKeypoints = kImages[image_index].getKeypoints();
	const cv::Matx34d kTargetProjectionMatrix = kImages[image_index].getProjectionMatrix();

	const int kPointNum = static_cast<int>(track.getTrackingNum());
	for (int i = 0; i < kPointNum; i++) {
		if (track.isRecoveredTriangulatedPoint(i)) {
			continue;
		}
		const int kTargetKeypointIndex = track.getTrackedKeypointIndex(i, image_index);
		if (kTargetKeypointIndex < 0) {
			continue;
		}
		const cv::Point2d kTargetKeypoint = kTargetKeypoints[kTargetKeypointIndex].pt;

		std::vector<cv::Matx34d> projection_matrix = { kTargetProjectionMatrix };
		std::vector<cv::Matx33d> rotation_matrix = { kImages[image_index].getRotationMatrix() };
		std::vector<cv::Matx31d> translation_vector = { kImages[image_index].getTranslation() };
		std::vector<cv::Point2d> image_points = { kTargetKeypoint };
		for (int j = 0; j < static_cast<int>(kImages.size()); j++) {
			if (!kImages[j].isRecoveredExtrinsicParameter() || j == image_index) {
				continue;
			}
			const std::vector<cv::KeyPoint>& kKeypoints = kImages[j].getKeypoints();
			const int kKeypointIndex = track.getTrackedKeypointIndex(i, j);
			if (kKeypointIndex < 0) {
				continue;
			}
			projection_matrix.push_back(kImages[j].getProjectionMatrix());
			rotation_matrix.push_back(kImages[j].getRotationMatrix());
			translation_vector.push_back(kImages[j].getTranslation());
			image_points.push_back(kKeypoints[kKeypointIndex].pt);
		}
		if (projection_matrix.size() >= 2) {
			cv::Point3d triangulated_point = CvUtil::triangulatePoints(image_points, projection_matrix);
			if (isInfinityPoint(kInfinityPointAngleDegree_, triangulated_point, rotation_matrix, translation_vector)) {
				std::cout << "Infinity" << std::endl;
				continue;
			}
			track.setTriangulatedPoint(i, triangulated_point.x, triangulated_point.y, triangulated_point.z);
		}
	}
}

bool SfM::isInfinityPoint(double degree_threshold, const cv::Point3d & kTriangulatedPoint, const std::vector<cv::Matx33d> & kRotationMatrix, const std::vector<cv::Matx31d> & kTranslationVector) const {
	if (kRotationMatrix.size() != kTranslationVector.size()) {
		return true;
	}

	std::vector<cv::Point3d> camera_positions(kRotationMatrix.size());
	for (size_t i = 0; i < kRotationMatrix.size(); i++) {
		camera_positions[i] = CvUtil::computeCameraPosition(kRotationMatrix[i], kTranslationVector[i]);
	}

	for (size_t i = 0; i < camera_positions.size(); i++) {
		for (size_t j = i + 1; j < camera_positions.size(); j++) {
			const double kAngleDegree = CvUtil::computeAngleDegree(camera_positions[i] - kTriangulatedPoint, camera_positions[j] - kTriangulatedPoint);
			if (kAngleDegree > degree_threshold) {
				return false;
			}
		}
	}

	return true;
}

bool SfM::removeHighReprojectionErrorTracks(Tracking & track, const std::vector<Image>& kImages) const {
	std::vector<std::vector<std::pair<double, int>>> reprojection_error_each_track(kImages.size());	// reprojection error, track index
	const int kTrackNum = static_cast<int>(track.getTrackingNum());
	for (int i = 0; i < kTrackNum; i++) {
		if (!track.isRecoveredTriangulatedPoint(i)) {
			continue;
		}

		const cv::Point3d& kTriangulatedPoint = track.getTriangulatedPoint(i);
		for (int j = 0; j < static_cast<int>(kImages.size()); j++) {
			if (!kImages[j].isRecoveredExtrinsicParameter()) {
				continue;
			}
			const int kKeypointIndex = track.getTrackedKeypointIndex(i, j);
			if (kKeypointIndex < 0) {
				continue;
			}
			const std::vector<cv::KeyPoint>& kKeypoints = kImages[j].getKeypoints();
			const cv::Point2d kImagePoint = kKeypoints[kKeypointIndex].pt;
			const cv::Matx34d kProjectionMatrix = kImages[j].getProjectionMatrix();
			const double kReprojectionError = CvUtil::computeReprojectionError(kImagePoint, kProjectionMatrix, kTriangulatedPoint);

			reprojection_error_each_track[j].push_back(std::make_pair(kReprojectionError, i));
		}
	}

	bool doRemoving = false;
	for (size_t i = 0; i < reprojection_error_each_track.size(); i++) {
		if (reprojection_error_each_track.at(i).size() == 0) {
			continue;
		}
		std::sort(reprojection_error_each_track[i].begin(), reprojection_error_each_track[i].end(),
			[](const std::pair<double, int>& x, const std::pair<double, int>& y) {
			return x.first < y.first;
		});
		const int kPercentileIndex = static_cast<int>(static_cast<double>(reprojection_error_each_track.at(i).size() * 0.8));
		const double kPercentileReprojectionError = reprojection_error_each_track.at(i).at(kPercentileIndex).first;
		const double kThreshold = MathUtil::clamp(kPercentileReprojectionError, 4.0, 16.0);
		std::cout << "Threshold: " << kThreshold << std::endl;

		for (int j = kPercentileIndex; j < reprojection_error_each_track.at(i).size(); j++) {
			if (reprojection_error_each_track.at(i).at(j).first > kThreshold) {
				track.removeTrack(reprojection_error_each_track.at(i).at(j).second);
				std::cout << "Remove index: " << reprojection_error_each_track.at(i).at(j).second << ": "<<reprojection_error_each_track.at(i).at(j).first<< std::endl;
				doRemoving = true;
			}
		}
	}

	return doRemoving;
}

void SfM::savePointCloud(const std::string & file_path) const {
	track_.saveTriangulatedPoints(file_path, images_);
}

void SfM::writeImageInfo(const std::string& dir_path) const {
	for (const auto& image : images_) {
		image.writeImageInfo(dir_path);
	}
}

void SfM::loadImageInfo(const std::string & dir_path) {
	const std::vector<std::experimental::filesystem::path> kFilePaths = FileUtil::readFiles(dir_path);
	for (const auto& kPath : kFilePaths) {
		Image image;
		std::cout << kPath.string() << std::endl;
		image.loadImageInfo(kPath.string());
		image.setFocalLength(kDefaultFocalLength_);
		images_.push_back(image);
	}
}

int SfM::selectInitialImagePair(const std::vector<Image>& kImages, const std::vector<ImagePair>& kImagePair) const {
	int initial_pair_index = 0;
	double initial_pair_possibility = 0.0;
	for (int i = 0; i < (int)kImagePair.size(); i++) {
		const std::array<int, 2> kImageIndex = kImagePair[i].getImageIndex();
		const cv::Size2i kImageSize1 = kImages.at(kImageIndex.at(0)).getImageSize();
		const cv::Size2i kImageSize2 = kImages.at(kImageIndex.at(1)).getImageSize();
		const int kMaxSize = std::max({kImageSize1.height, kImageSize1.width, kImageSize2.height, kImageSize2.width});
		
		double baseline_possibility = kImagePair[i].computeBaeslinePossibility(kImages.at(kImageIndex.at(0)), kImages.at(kImageIndex.at(1)), (double)kMaxSize*kHomographyThresholdRatio_*0.01);
		if (baseline_possibility > initial_pair_possibility && kImagePair[i].getMatchNum()>kMinimumInitialImagePairNum_) {
			initial_pair_index = i;
		}
	}

	return initial_pair_index;
}

void SfM::optimization(Tracking& track, std::vector<Image>& images) const {
	BundleAdjustment bundle_adjustment;
	bundle_adjustment.runBundleAdjustment(images, track);
}
