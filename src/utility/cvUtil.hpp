#ifndef CV_UTIL_HPP
#define CV_UTIL_HPP

#include <opencv2/opencv.hpp>

class CvUtil {
public:
	static cv::Point3d convertImagePointToCameraVector(const cv::Point2d& kImagePoint, const cv::Matx33d& kIntrinsicParameter);
	static cv::Point3d computeCameraPosition(const cv::Matx33d& kRotationMatrix, const cv::Matx31d& kTranslationVector);
	static cv::Point3d computeCameraPosition(const cv::Matx34d& kExtrinsicParameter);
	static cv::Point2d computeProjectionPoint(const cv::Matx34d& kProjectionMatrix, const cv::Point3d& kWorldPoint);
	static double computeReprojectionError(const cv::Point2d& kImagePoint, const cv::Matx34d& kProjectionMatrix, const cv::Point3d& kWorldPoint);
	static cv::Matx34d computeProjectionMatrix(const std::vector<cv::Point2d>& kImagePoints, const std::vector<cv::Point3d>& kWorldPoints);
	static cv::Matx34d computeProjectionMatrixUsingRansac(const std::vector<cv::Point2d>& kImagePoints, const std::vector<cv::Point3d>& kWorldPoints, double threshold, double inlier_ratio, double probability = 0.99);
	static bool decomposeProjectionMatrix(const cv::Matx34d& kProjectionMatrix, cv::Matx33d& intrinsic_parameter, cv::Matx33d& rotation_matrix, cv::Matx31d& translation_vector);
	static cv::Point3d triangulatePoints(const std::vector<cv::Point2d>& kImagePoints, const std::vector<cv::Matx34d>& kProjectionMatrix);
	static double computeAngleRadian(const cv::Vec3d& kVec1, const cv::Vec3d& kVec2);
	static double computeAngleDegree(const cv::Vec3d& kVec1, const cv::Vec3d& kVec2);
	static double computeAngleDegree(const cv::Matx31d& kVec1, const cv::Matx31d& kVec2);
	static double readFocalLengthFromExif(const std::string& kImagePath);

private:
	static int computeRansacIterationNum(int select_num, double probability, double inlier_ratio);
	static std::vector<int> selectRandomNElements(int element_num, int select_num, int seed);
	static int countProjectionMatrixInlier(const cv::Matx34d& kProjectionMatrix, const std::vector<cv::Point2d>& kImagePoints, const std::vector<cv::Point3d>& kWorldPoints, double threshold);
};

#endif