#include "sfm.hpp"

SfM::SfM() : kDetectorType(Image::DetectorType::SIFT), kMinimumInitialImagePairNum(100) {
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
		images_.push_back(image);
		std::cout << " done." << std::endl;
	}
}

void SfM::detectKeypoints() {
	for (auto& image : images_) {
		image.detectKeyPoints(kDetectorType);
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

int SfM::selectInitialImagePair(const std::vector<Image>& images, const std::vector<ImagePair>& image_pair) const {
	int initial_pair_index = 0;
	double initial_pair_possibility = 0.0;
	for (int i = 0; i < (int)image_pair.size(); i++) {
		const std::array<int, 2> kImageIndex = image_pair[i].getImageIndex();
		double baseline_possibility = image_pair[i].computeBaeslinePossibility(images.at(kImageIndex.at(0)), images.at(kImageIndex.at(1)));
		if (baseline_possibility > initial_pair_possibility && image_pair[i].getMatchNum()>kMinimumInitialImagePairNum) {
			initial_pair_index = i;
		}
	}

	return initial_pair_index;
}
