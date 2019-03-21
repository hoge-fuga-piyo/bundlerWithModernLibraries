#include "sfm.hpp"
#include "bundleAdjustment.hpp"
#include "cvUtil.hpp"

SfM::SfM() : kDetectorType_(Image::DetectorType::SIFT), kMinimumInitialImagePairNum_(100), kHomographyThresholdRatio_(0.4), kDefaultFocalLength_(532.0){
}

void SfM::loadImages(const std::string kDirPath) {
	const std::vector<std::experimental::filesystem::path> kFilePaths = FileUtil::readFiles(kDirPath);
	for (const auto& kPath : kFilePaths) {
		const std::string kExtension = kPath.extension().string();
		if (kExtension != ".jpg" && kExtension != ".JPG" && kExtension != ".png" && kExtension != ".PNG") {
			continue;
		}
		std::cout << "Load " << kPath.string() << "...";
		Image image;
		image.loadImage(kPath.string());
		image.setFocalLength(kDefaultFocalLength_);
		images_.push_back(image);
		std::cout << " done." << std::endl;
	}
}

void SfM::detectKeypoints() {
	for (auto& image : images_) {
		image.detectKeyPoints(kDetectorType_);
	}
}

void SfM::keypointMatching() {
	for (int i = 0; i < images_.size(); i++) {
		for (int j = i + 1; j < images_.size(); j++) {
			ImagePair image_pair;
			image_pair.setImageIndex(i, j);
			image_pair.keypointMatching(images_[i], images_[j]);
			//image_pair.showMatches(images_[i].getImage(), images_[i].getKeypoints(), images_[j].getImage(), images_[j].getKeypoints());
			image_pair_.push_back(image_pair);
		}
	}
	std::cout << image_pair_.size() << " image pairs are found." << std::endl;
}

void SfM::trackingKeypoint() {
	track_.tracking((int)images_.size(), image_pair_);
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
	optimization(track_, images_);
}

bool SfM::nextReconstruct() {
	int next_image_index = selectNextReconstructImage(track_, images_);
	std::cout <<"Next image index:" <<next_image_index << std::endl;

	std::vector<cv::Point2d> image_points;
	std::vector<cv::Point3d> world_points;
	track_.extractImagePointAndWorlPointPairs(next_image_index, images_[next_image_index], image_points, world_points);
	const cv::Mat& kImage = images_[next_image_index].getImage();
	const double threshold = std::max(kImage.cols, kImage.rows) * 0.004;
	std::cout << "threshold: " << threshold << std::endl;
	const cv::Matx34d kCameraParam = CvUtil::computeCameraParameterUsingRansac(image_points, world_points, threshold, 0.5, 0.9999);
	std::cout << "cameraparam" << std::endl;
	std::cout << kCameraParam << std::endl;

	cv::Matx33d intrinsic_param;
	cv::Matx33d rotation_mat;
	cv::Matx41d homogeneous_camera_position;
	cv::decomposeProjectionMatrix(kCameraParam, intrinsic_param, rotation_mat, homogeneous_camera_position);
	cv::Matx31d camera_position(homogeneous_camera_position(0) / homogeneous_camera_position(3)
							, homogeneous_camera_position(1) / homogeneous_camera_position(3)
							, homogeneous_camera_position(2) / homogeneous_camera_position(3));
	cv::Matx31d translation_vec = -rotation_mat * camera_position;

	cv::Matx34d extrinsic_param(rotation_mat(0, 0), rotation_mat(0, 1), rotation_mat(0, 2), translation_vec(0)
							, rotation_mat(1, 0), rotation_mat(1, 1), rotation_mat(1, 2), translation_vec(1)
							, rotation_mat(2, 0), rotation_mat(2, 1), rotation_mat(2, 2), translation_vec(2));

	images_[next_image_index].setExtrinsicParameter(rotation_mat, translation_vec);
	images_[next_image_index].setFocalLength((intrinsic_param(0, 0) + intrinsic_param(1, 1)) / 2.0);
	images_[next_image_index].setPrincipalPoint(intrinsic_param(0, 2), intrinsic_param(1, 2));

	std::cout << "re-cameraparam" << std::endl;
	std::cout << intrinsic_param * extrinsic_param << std::endl;
	std::cout << "intrinsic" << std::endl;
	std::cout << intrinsic_param << std::endl;

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
			next_image_index = i;
		}
	}
	return next_image_index;
}

void SfM::savePointCloud(const std::string & file_path) const {
	track_.saveTriangulatedPoints(file_path, images_);
}

int SfM::selectInitialImagePair(const std::vector<Image>& kImages, const std::vector<ImagePair>& kImagePair) const {
	int initial_pair_index = 0;
	double initial_pair_possibility = 0.0;
	for (int i = 0; i < (int)kImagePair.size(); i++) {
		const std::array<int, 2> kImageIndex = kImagePair[i].getImageIndex();
		const cv::Size2i kImageSize1 = kImages.at(kImageIndex.at(0)).getImage().size();
		const cv::Size2i kImageSize2 = kImages.at(kImageIndex.at(1)).getImage().size();
		const int kMaxSize = std::max({kImageSize1.height, kImageSize1.width, kImageSize2.height, kImageSize2.width});
		
		double baseline_possibility = kImagePair[i].computeBaeslinePossibility(kImages.at(kImageIndex.at(0)), kImages.at(kImageIndex.at(1)), (double)kMaxSize*kHomographyThresholdRatio_*0.01);
		if (baseline_possibility > initial_pair_possibility && kImagePair[i].getMatchNum()>kMinimumInitialImagePairNum_) {
			initial_pair_index = i;
		}
	}

	return initial_pair_index;
}

void SfM::optimization(Tracking& track, std::vector<Image>& images) const {
	std::cout << "Start optimization" << std::endl;
	BundleAdjustment bundle_adjustment;
	bundle_adjustment.runBundleAdjustment(images, track);
	std::cout << "Finish optimization" << std::endl;
}

