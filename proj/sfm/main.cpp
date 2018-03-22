#include <iostream>
#include <opencv2/opencv.hpp>
#include "image.hpp"
#include "image_pair.hpp"
#include "fileUtil.hpp"

int main(){
	std::string img_dir = "../../../sampledata/fountain_int";
	std::vector<std::experimental::filesystem::path> paths = FileUtil::readFiles(img_dir);

	std::vector<Image> images;
	for (const auto& path : paths) {
		std::cout << path.string()<<std::endl;
		Image image;
		image.loadImage(path.string());
		image.detectKeyPoints(Image::DetectorType::SIFT);
		images.push_back(image);
	}

	ImagePair image_pair;
	image_pair.setImageIndex(0, 1);
	const std::vector<cv::KeyPoint> keypoint1 = images[0].getKeypoints();
	const std::vector<cv::KeyPoint> keypoint2 = images[4].getKeypoints();
	image_pair.keypointMatching(images[0], images[4]);
	image_pair.showMatches(images[0].getImage(), keypoint1, images[4].getImage(), keypoint2);
	
	return 0;
}