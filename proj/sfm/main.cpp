#include <iostream>
#include <opencv2/opencv.hpp>
#include "sfm.hpp"

int main(int args, char** argv){
	google::InitGoogleLogging(argv[0]);

	SfM sfm;
	sfm.loadImages("../../../sampledata/fountain_int");
	sfm.detectKeypoints();
	sfm.keypointMatching();
	sfm.trackingKeypoint();
	sfm.initialReconstruct();

	sfm.nextReconstruct();

	sfm.savePointCloud("result.ply");

	//std::string img_dir = "../../../sampledata/fountain_int";
	//std::vector<std::experimental::filesystem::path> paths = FileUtil::readFiles(img_dir);

	//std::vector<Image> images;
	//for (const auto& path : paths) {
	//	std::cout << path.extension() << std::endl;
	//	std::cout << path.string()<<std::endl;
	//	Image image;
	//	image.loadImage(path.string());
	//	image.detectKeyPoints(Image::DetectorType::SIFT);
	//	images.push_back(image);
	//}

	//ImagePair image_pair;
	//image_pair.setImageIndex(0, 1);
	//const std::vector<cv::KeyPoint> keypoint1 = images[0].getKeypoints();
	//const std::vector<cv::KeyPoint> keypoint2 = images[4].getKeypoints();
	//image_pair.keypointMatching(images[0], images[4]);
	//image_pair.showMatches(images[0].getImage(), keypoint1, images[4].getImage(), keypoint2);
	//
	return 0;
}