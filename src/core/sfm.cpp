#include "sfm.hpp"

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
