#ifndef CV_UTIL_HPP
#define CV_UTIL_HPP

#include <opencv2/opencv.hpp>

class CvUtil {
public:
	static cv::Point3d convertImagePointToCameraVector(const cv::Point2d& kImagePoint, const cv::Matx33d& kIntrinsicParameter);
	static cv::Matx34d computeCameraParameter(const std::vector<cv::Point2d>& kImagePoints, const std::vector<cv::Point3d>& kWorldPoints);
	static cv::Matx34d computeCameraParameterUsingRansac(const std::vector<cv::Point2d>& kImagePoints, const std::vector<cv::Point3d>& kWorldPoints, double threshold, double inlier_ratio, double probability = 0.99);
private:
	static int computeRansacIterationNum(int select_num, double probability, double inlier_ratio);
	static std::vector<int> selectRandomNElements(int element_num, int select_num, int seed);
	static int countCameraParameterInlier(const cv::Matx34d& kCameraParam, const std::vector<cv::Point2d>& kImagePoints, const std::vector<cv::Point3d>& kWorldPoints, double threshold);
};

#endif