#ifndef CV_UTIL_HPP
#define CV_UTIL_HPP

#include <opencv2/opencv.hpp>

class CvUtil {
public:
	static cv::Point3d convertImagePointToCameraVector(const cv::Point2d& kImagePoint, const cv::Matx33d& kIntrinsicParameter);
	static cv::Matx34d computeCameraParameter(const std::vector<cv::Point2d>& kImagePoints, const std::vector<cv::Point3d>& kWorldPoints);
private:

};

#endif