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

	SfM sfm;
	if (kUseImageInfoLogs) {
		sfm.loadImageInfo("./logs");
	} else {
		//sfm.loadImagesAndDetectKeypoints("../../../sampledata/fountain_images");
		//sfm.loadImagesAndDetectKeypoints("../../../sampledata/herzjesu_dense");
		//sfm.loadImagesAndDetectKeypoints("../../../sampledata/castle_dense");

		sfm.loadImagesAndDetectKeypoints("../../../sampledata/fountain_int");

		//sfm.loadImagesAndDetectKeypoints("../../../sampledata/NotreDame");
		//sfm.loadImages("../../../sampledata/NotreDame");

		sfm.writeImageInfo("./logs");
	}

	sfm.keypointMatching();
	sfm.trackingKeypoint();
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