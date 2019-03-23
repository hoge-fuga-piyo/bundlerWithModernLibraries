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

	sfm.nextReconstruct();

	sfm.savePointCloud("result_next.ply");

	return 0;
}