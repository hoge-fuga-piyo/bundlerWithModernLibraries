#include <iostream>
#include <opencv2/opencv.hpp>
#include "sfm.hpp"

int main(int args, char** argv){
#ifdef _OPENMP
	std::cout << "OpenMP is valid" << std::endl;
#else
	std::cout << "OpenMP is invalid" << std::endl;
#endif

	google::InitGoogleLogging(argv[0]);

	const bool kUseImageInfoLogs = true;
	const bool kUseImagePairLogs = true;
	const bool kUseTrackingLogs = true;

	const std::string kSampleData = "fountain_int";

	const std::string kImageInfoLogDir = "./logs/" + kSampleData + "/image";
	const std::string kImagePairInfoLogDir = "./logs/" + kSampleData + "/image_pair";
	const std::string kTrackingInfoLogDir = "./logs/" + kSampleData + "/tracking";

	SfM sfm;

	// detect keypoints
	if (kUseImageInfoLogs) {
		sfm.loadImageInfo(kImageInfoLogDir);
	} else {
		sfm.loadImagesAndDetectKeypoints("../../../sampledata/" + kSampleData);

		//sfm.loadImagesAndDetectKeypoints("../../../sampledata/fountain_images");
		//sfm.loadImagesAndDetectKeypoints("../../../sampledata/herzjesu_dense");
		//sfm.loadImagesAndDetectKeypoints("../../../sampledata/castle_dense");

		//sfm.loadImagesAndDetectKeypoints("../../../sampledata/fountain_int");

		//sfm.loadImagesAndDetectKeypoints("../../../sampledata/NotreDame");
		//sfm.loadImagesAndDetectKeypoints("../../../sampledata/NotreDame");

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
		//const std::string kSaveNamePreffix = "result_notredame";
		sfm.savePointCloud(kSaveNamePreffix + std::to_string(i) + ".ply");
		i++;
	}

	//sfm.nextReconstruct();

	sfm.savePointCloud("result_all.ply");

	return 0;
}