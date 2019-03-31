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

	sfm.savePointCloud("result_init.ply");

	int i = 0;
	while (sfm.nextReconstruct()) {
		const std::string kSaveNamePreffix = "result_";
		sfm.savePointCloud(kSaveNamePreffix + std::to_string(i) + ".ply");
		i++;
	}
	//sfm.nextReconstruct();

	sfm.savePointCloud("result_all.ply");

	return 0;
}