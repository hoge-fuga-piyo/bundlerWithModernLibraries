#include "cvUtil.hpp"

cv::Point3d CvUtil::convertImagePointToCameraVector(const cv::Point2d & kImagePoint, const cv::Matx33d & kIntrinsicParameter) {
	cv::Point3d camera_vec;
	camera_vec.x = (kImagePoint.x - kIntrinsicParameter(0, 2)) / kIntrinsicParameter(0, 0);
	camera_vec.y = (kImagePoint.y - kIntrinsicParameter(1, 2)) / kIntrinsicParameter(1, 1);
	camera_vec.z = 1.0;
	return camera_vec;
}
