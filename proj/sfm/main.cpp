#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "sfm.hpp"

int main(int args, char** argv){
#ifdef _OPENMP
	std::cout << "OpenMP is valid" << std::endl;
#else
	std::cout << "OpenMP is invalid" << std::endl;
#endif

	google::InitGoogleLogging(argv[0]);

	const bool kUseImageInfoLogs = false;
	const bool kUseImagePairLogs = false;
	const bool kUseTrackingLogs = false;

	//const std::string kSampleData = "fountain_int";
	//const std::string kSampleData = "fountain_images";
	//const std::string kSampleData = "herzjesu_dense";
	//const std::string kSampleData = "castle_dense";
	const std::string kSampleData = "herzjesu_dense_large";
	//const std::string kSampleData = "NotreDame";

	const std::string kImageInfoLogDir = "./logs/" + kSampleData + "/image";
	const std::string kImagePairInfoLogDir = "./logs/" + kSampleData + "/image_pair";
	const std::string kTrackingInfoLogDir = "./logs/" + kSampleData + "/tracking";

	std::filesystem::create_directories(kImageInfoLogDir);
	std::filesystem::create_directories(kImagePairInfoLogDir);
	std::filesystem::create_directories(kTrackingInfoLogDir);

	SfM sfm;

	// detect keypoints
	if (kUseImageInfoLogs) {
		sfm.loadImageInfo(kImageInfoLogDir);
	} else {
		sfm.loadImagesAndDetectKeypoints("../../../sampledata/" + kSampleData);
		sfm.writeImageInfo(kImageInfoLogDir);
	}

	// keypoint matching
	if (kUseImagePairLogs) {
		sfm.loadImagePairInfo(kImagePairInfoLogDir);
	} else {
		sfm.keypointMatching();
		sfm.writeImagePairInfo(kImagePairInfoLogDir);
	}

	// keypoint tracking
	if (kUseTrackingLogs) {
		sfm.loadTrackingInfo(kTrackingInfoLogDir);
	} else {
		sfm.trackingKeypoint();
		sfm.writeTrackingInfo(kTrackingInfoLogDir);
	}

	sfm.initialReconstruct();

	sfm.savePointCloud("result_init.ply");

	int i = 0;
	while (sfm.nextReconstruct()) {
		const std::string kSaveNamePreffix = "result_";
		sfm.savePointCloud(kSaveNamePreffix + std::to_string(i) + ".ply");
		i++;
	}

	sfm.savePointCloud("result_all.ply");

	return 0;
}