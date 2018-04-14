#include "sfm.hpp"

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
	track_.tracking(images_.size(), image_pair_);
	std::cout << "Tracking num: " << track_.getTrackingNum() << std::endl;
}
