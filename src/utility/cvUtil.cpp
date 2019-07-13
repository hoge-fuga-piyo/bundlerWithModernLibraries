#include "cvUtil.hpp"
#include "mathUtil.hpp"
#include <random>
#include <cmath>

cv::Point3d CvUtil::convertImagePointToCameraVector(const cv::Point2d & kImagePoint, const cv::Matx33d & kIntrinsicParameter) {
	cv::Point3d camera_vec;
	camera_vec.x = (kImagePoint.x - kIntrinsicParameter(0, 2)) / kIntrinsicParameter(0, 0);
	camera_vec.y = (kImagePoint.y - kIntrinsicParameter(1, 2)) / kIntrinsicParameter(1, 1);
	camera_vec.z = 1.0;
	return camera_vec;
}

cv::Point3d CvUtil::computeCameraPosition(const cv::Matx33d & rotation_matrix, const cv::Matx31d translation_vector) {
	const cv::Matx31d kCameraPosition = -rotation_matrix.t()*translation_vector;
	return cv::Point3d(kCameraPosition(0), kCameraPosition(1), kCameraPosition(2));
}

cv::Point3d CvUtil::computeCameraPosition(const cv::Matx34d & extrinsic_parameter) {
	cv::Matx33d rotation_matrix(extrinsic_parameter(0, 0), extrinsic_parameter(0, 1), extrinsic_parameter(0, 2)
		, extrinsic_parameter(1, 0), extrinsic_parameter(1, 1), extrinsic_parameter(1, 2)
		, extrinsic_parameter(2, 0), extrinsic_parameter(2, 1), extrinsic_parameter(2, 2));
	cv::Matx31d translation_vector = extrinsic_parameter.col(3);

	return computeCameraPosition(rotation_matrix, translation_vector);
}

cv::Point2d CvUtil::computeProjectionPoint(const cv::Matx34d & kProjectionMatrix, const cv::Point3d & kWorldPoint) {
	const cv::Matx41d kHomogeneousPoint(kWorldPoint.x, kWorldPoint.y, kWorldPoint.z, 1.0);
	const cv::Matx31d kProjectionPoint = kProjectionMatrix * kHomogeneousPoint;

	return cv::Point2d(kProjectionPoint(0) / kProjectionPoint(2), kProjectionPoint(1) / kProjectionPoint(2));
}

double CvUtil::computeReprojectionError(const cv::Point2d & kImagePoint, const cv::Matx34d & kProjectionMatrix, const cv::Point3d & kWorldPoint) {
	const cv::Point2d kReprojectionPoint = computeProjectionPoint(kProjectionMatrix, kWorldPoint);
	return cv::norm(kImagePoint - kReprojectionPoint);
}

cv::Matx34d CvUtil::computeCameraParameter(const std::vector<cv::Point2d>& kImagePoints, const std::vector<cv::Point3d>& kWorldPoints) {
	if (kImagePoints.size() != kWorldPoints.size()) {
		throw std::invalid_argument("[ERROR] The number of image points and world points are different.");
	}

	if (kImagePoints.size() < 6) {
		throw std::invalid_argument("[ERROR] The number of points is not enough.");
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

	const cv::Mat kMinSingularVector = vt.row(vt.rows - 1);
	const cv::Matx34d kCameraPraram(kMinSingularVector.at<double>(0, 0), kMinSingularVector.at<double>(0, 1), kMinSingularVector.at<double>(0, 2), kMinSingularVector.at<double>(0, 3)
		, kMinSingularVector.at<double>(0, 4), kMinSingularVector.at<double>(0, 5), kMinSingularVector.at<double>(0, 6), kMinSingularVector.at<double>(0, 7)
		, kMinSingularVector.at<double>(0, 8), kMinSingularVector.at<double>(0, 9), kMinSingularVector.at<double>(0, 10), kMinSingularVector.at<double>(0, 11));

	return kCameraPraram;
}

cv::Matx34d CvUtil::computeCameraParameterUsingRansac(const std::vector<cv::Point2d>& kImagePoints, const std::vector<cv::Point3d>& kWorldPoints, double threshold, double inlier_ratio, double probability) {
	if (kImagePoints.size() != kWorldPoints.size()) {
		std::cout << "[ERROR] The number of image points and world points are different." << std::endl;
		return cv::Matx34d();
	}
	if (kImagePoints.size() < 6) {
		std::cout << "[ERROR] The number of points is not enough." << std::endl;
		return cv::Matx34d();
	}

	int iteration_num = computeRansacIterationNum(6, probability, inlier_ratio);
	std::cout << "Iteration Num: " << iteration_num << std::endl;

	int max_inlier_num = 0;
	cv::Matx34d best_camera_param = cv::Matx34d::zeros();
	for (int i = 0; i < iteration_num; i++) {
		std::vector<int> random_point_indexes = selectRandomNElements(static_cast<int>(kImagePoints.size()), 6, i);
		std::vector<cv::Point2d> random_image_points(6);
		std::vector<cv::Point3d> random_world_points(6);
		for (int j = 0; j < 6; j++) {
			random_image_points[j] = kImagePoints[random_point_indexes[j]];
			random_world_points[j] = kWorldPoints[random_point_indexes[j]];
		}

		const cv::Matx34d kCameraParam = computeCameraParameter(random_image_points, random_world_points);
		int inlier_num = countCameraParameterInlier(kCameraParam, kImagePoints, kWorldPoints, threshold);
		if (inlier_num > max_inlier_num) {
			max_inlier_num = inlier_num;
			best_camera_param = kCameraParam;
		}
	}
	std::cout << "max inlier: " << max_inlier_num << std::endl;

	return std::move(best_camera_param);
}

bool CvUtil::decomposeProjectionMatrix(const cv::Matx34d & kProjectionMatrix, cv::Matx33d & intrinsic_parameter, cv::Matx33d & rotation_matrix, cv::Matx31d & translation_vector) {
	cv::Matx33d decomp_mat(kProjectionMatrix(0, 0), kProjectionMatrix(0, 1), kProjectionMatrix(0, 2)
		, kProjectionMatrix(1, 0), kProjectionMatrix(1, 1), kProjectionMatrix(1, 2)
		, kProjectionMatrix(2, 0), kProjectionMatrix(2, 1), kProjectionMatrix(2, 2));
	cv::RQDecomp3x3(decomp_mat, intrinsic_parameter, rotation_matrix);
	const double kScale = 1.0 / intrinsic_parameter(2, 2);
	intrinsic_parameter *= kScale;
	cv::Matx33d sign_matrix = cv::Matx33d::eye();
	if (intrinsic_parameter(0, 0) < 0) {
		if (intrinsic_parameter(1, 1) > 0) {
			return false;
		}
		sign_matrix(0, 0) *= -1.0;
		sign_matrix(1, 1) *= -1.0;
	}
	intrinsic_parameter = intrinsic_parameter * sign_matrix;
	rotation_matrix = sign_matrix * rotation_matrix;
	translation_vector = kScale * intrinsic_parameter.inv() * kProjectionMatrix.col(3);

	return true;
}

cv::Point3d CvUtil::triangulatePoints(const std::vector<cv::Point2d>& kImagePoints, const std::vector<cv::Matx34d>& kProjectionMatrix) {
	if (kImagePoints.size() != kProjectionMatrix.size()) {
		return cv::Point3d();
	}

	if (kProjectionMatrix.size() < 2) {
		return cv::Point3d();
	}

	cv::Mat mat = cv::Mat_<double>(kProjectionMatrix.size() * 2, 4);
	for (size_t i = 0; i < kProjectionMatrix.size(); i++) {
		const cv::Matx14d kCol1 = kImagePoints[i].x * kProjectionMatrix[i].row(2) - kProjectionMatrix[i].row(0);
		const cv::Matx14d kCol2 = kImagePoints[i].y * kProjectionMatrix[i].row(2) - kProjectionMatrix[i].row(1);

		mat.at<double>(i * 2 + 0, 0) = kCol1(0);
		mat.at<double>(i * 2 + 0, 1) = kCol1(1);
		mat.at<double>(i * 2 + 0, 2) = kCol1(2);
		mat.at<double>(i * 2 + 0, 3) = kCol1(3);
		mat.at<double>(i * 2 + 1, 0) = kCol2(0);
		mat.at<double>(i * 2 + 1, 1) = kCol2(1);
		mat.at<double>(i * 2 + 1, 2) = kCol2(2);
		mat.at<double>(i * 2 + 1, 3) = kCol2(3);
	}

	cv::Mat w, u, vt;
	cv::SVD::compute(mat, w, u, vt, cv::SVD::MODIFY_A);

	const cv::Mat kMinSingularVector = vt.row(vt.rows - 1);
	cv::Point3d triangulated_point(kMinSingularVector.at<double>(0, 0) / kMinSingularVector.at<double>(0, 3)
		, kMinSingularVector.at<double>(0, 1) / kMinSingularVector.at<double>(0, 3)
		, kMinSingularVector.at<double>(0, 2) / kMinSingularVector.at<double>(0, 3));

	return triangulated_point;
}

double CvUtil::computeAngleRadian(const cv::Vec3d kVec1, const cv::Vec3d kVec2) {
	double cos = kVec1.dot(kVec2) / (cv::norm(kVec1, cv::NORM_L2) * cv::norm(kVec2, cv::NORM_L2));
	double radian = std::acos(cos);
	return radian;
}

double CvUtil::computeAngleDegree(const cv::Vec3d kVec1, const cv::Vec3d kVec2) {
	double radian = computeAngleRadian(kVec1, kVec2);
	return MathUtil::convertRadianToDegree(radian);
}

double CvUtil::computeAngleDegree(const cv::Matx31d kVec1, const cv::Matx31d kVec2) {
	const cv::Vec3d kVec1_vec(kVec1(0), kVec1(1), kVec1(2));
	const cv::Vec3d kVec2_vec(kVec2(0), kVec2(1), kVec2(2));

	return computeAngleDegree(kVec1_vec, kVec2_vec);
}

int CvUtil::computeRansacIterationNum(int select_num, double probability, double inlier_ratio) {
	double all_inlier_probability = std::pow(inlier_ratio, select_num);
	double at_least_one_outlier_probability = 1.0 - all_inlier_probability;
	double iteration_num = std::log(1 - probability) / std::log(at_least_one_outlier_probability);

	return static_cast<int>(std::ceil(iteration_num));
}

std::vector<int> CvUtil::selectRandomNElements(int element_num, int select_num, int seed) {
	std::mt19937 rand(seed);
	std::uniform_int_distribution<int> uniform_rand(0, element_num - 1);

	std::vector<int> selected_index;
	for (int i = 0; i < select_num; i++) {
		int rand_index = uniform_rand(rand);

		// Avoid selecting duplication elements
		bool is_duplication = false;
		for (size_t j = 0; j < selected_index.size(); j++) {
			if (selected_index[i] == rand_index) {
				is_duplication = true;
				break;
			}
		}
		if (is_duplication) {
			i--;
			continue;
		}

		selected_index.push_back(rand_index);
	}
	return std::move(selected_index);
}

int CvUtil::countCameraParameterInlier(const cv::Matx34d & kCameraParam, const std::vector<cv::Point2d>& kImagePoints, const std::vector<cv::Point3d>& kWorldPoints, double threshold) {
	int inlier_num = 0;
	for (size_t i = 0; i < kImagePoints.size(); i++) {
		const cv::Matx41d kWorldPoint(kWorldPoints[i].x, kWorldPoints[i].y, kWorldPoints[i].z, 1.0);
		const cv::Matx31d kTmpReprojectionPoint = kCameraParam * kWorldPoint;
		const cv::Point2d kReprojectionPoint(kTmpReprojectionPoint(0) / kTmpReprojectionPoint(2), kTmpReprojectionPoint(1) / kTmpReprojectionPoint(2));
		double distance = cv::norm(kImagePoints[i] - kReprojectionPoint);
		if (distance < threshold) {
			inlier_num++;
		}
	}

	return inlier_num;
}
