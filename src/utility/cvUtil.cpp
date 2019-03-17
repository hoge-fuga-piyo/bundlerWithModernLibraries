#include "cvUtil.hpp"

cv::Point3d CvUtil::convertImagePointToCameraVector(const cv::Point2d & kImagePoint, const cv::Matx33d & kIntrinsicParameter) {
	cv::Point3d camera_vec;
	camera_vec.x = (kImagePoint.x - kIntrinsicParameter(0, 2)) / kIntrinsicParameter(0, 0);
	camera_vec.y = (kImagePoint.y - kIntrinsicParameter(1, 2)) / kIntrinsicParameter(1, 1);
	camera_vec.z = 1.0;
	return camera_vec;
}

cv::Matx34d CvUtil::computeCameraParameter(const std::vector<cv::Point2d>& kImagePoints, const std::vector<cv::Point3d>& kWorldPoints) {
	if (kImagePoints.size() != kWorldPoints.size()) {
		std::cout << "[ERROR] The number of image points and world points are different." << std::endl;
		return cv::Matx34d();
	}
	
	cv::Mat mat = cv::Mat::zeros(kImagePoints.size() * 2, 12, CV_64FC1);
	for (size_t i = 0; i < kImagePoints.size(); i++) {
		mat.at<double>(i * 2 + 0, 0) = kWorldPoints[i].x;
		mat.at<double>(i * 2 + 0, 1) = kWorldPoints[i].y;
		mat.at<double>(i * 2 + 0, 2) = kWorldPoints[i].z;
		mat.at<double>(i * 2 + 0, 3) = 1.0;
		mat.at<double>(i * 2 + 0, 8) = -kImagePoints[i].x * kWorldPoints[i].x;
		mat.at<double>(i * 2 + 0, 9) = -kImagePoints[i].x * kWorldPoints[i].y;
		mat.at<double>(i * 2 + 0, 10) = -kImagePoints[i].x * kWorldPoints[i].z;
		mat.at<double>(i * 2 + 0, 11) = -kImagePoints[i].x;

		mat.at<double>(i * 2 + 1, 4) = kWorldPoints[i].x;
		mat.at<double>(i * 2 + 1, 5) = kWorldPoints[i].y;
		mat.at<double>(i * 2 + 1, 6) = kWorldPoints[i].z;
		mat.at<double>(i * 2 + 1, 7) = 1.0;
		mat.at<double>(i * 2 + 1, 8) = -kImagePoints[i].y * kWorldPoints[i].x;
		mat.at<double>(i * 2 + 1, 9) = -kImagePoints[i].y * kWorldPoints[i].y;
		mat.at<double>(i * 2 + 1, 10) = -kImagePoints[i].y * kWorldPoints[i].z;
		mat.at<double>(i * 2 + 1, 11) = -kImagePoints[i].y;
	}

	cv::Mat w, u, vt;
	cv::SVD::compute(mat, w, u, vt, cv::SVD::MODIFY_A);

	std::cout << vt << std::endl;

	const cv::Mat kMinSingularVector = vt.row(vt.rows - 1);
	const cv::Matx34d kCameraPraram(kMinSingularVector.at<double>(0, 0), kMinSingularVector.at<double>(0, 1), kMinSingularVector.at<double>(0, 2), kMinSingularVector.at<double>(0, 3)
		, kMinSingularVector.at<double>(0, 4), kMinSingularVector.at<double>(0, 5), kMinSingularVector.at<double>(0, 6), kMinSingularVector.at<double>(0, 7)
		, kMinSingularVector.at<double>(0, 8), kMinSingularVector.at<double>(0, 9), kMinSingularVector.at<double>(0, 10), kMinSingularVector.at<double>(0, 11));

	std::cout << kCameraPraram << std::endl;

	return kCameraPraram;
}
