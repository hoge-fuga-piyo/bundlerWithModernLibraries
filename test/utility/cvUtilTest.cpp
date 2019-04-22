#include <gtest/gtest.h>
#include <limits>
#include "cvUtil.hpp"

TEST(CvUtilTest, convertImagePointToCameraVector) {
	{
		const cv::Point2d kImagePoint(5.0, 10.0);
		const cv::Matx33d kIntrinsicParameter(100.0, 0.0, 500.0
			, 0.0, 100.0, 400.0
			, 0.0, 0.0, 1.0);
		const cv::Point3d kCameraVector = CvUtil::convertImagePointToCameraVector(kImagePoint, kIntrinsicParameter);
		EXPECT_DOUBLE_EQ(-4.95, kCameraVector.x);
		EXPECT_DOUBLE_EQ(-3.9, kCameraVector.y);
		EXPECT_DOUBLE_EQ(1.0, kCameraVector.z);
	}

	{
		const cv::Point2d kImagePoint(0.0, 0.0);
		const cv::Matx33d kIntrinsicParameter(100.0, 0.0, 500.0
			, 0.0, 100.0, 400.0
			, 0.0, 0.0, 1.0);
		const cv::Point3d kCameraVector = CvUtil::convertImagePointToCameraVector(kImagePoint, kIntrinsicParameter);
		EXPECT_DOUBLE_EQ(-5.0, kCameraVector.x);
		EXPECT_DOUBLE_EQ(-4.0, kCameraVector.y);
		EXPECT_DOUBLE_EQ(1.0, kCameraVector.z);
	}
}

TEST(CvUtilTest, computeCameraPosition1) {
	{
		const cv::Matx33d kRotationMatrix(1.0, 0.0, 0.0
			, 0.0, 1.0, 0.0
			, 0.0, 0.0, 1.0);
		const cv::Matx31d kTranslation(1.0, 1.0, 1.0);
		const cv::Point3d kCameraPosition = CvUtil::computeCameraPosition(kRotationMatrix, kTranslation);
		EXPECT_DOUBLE_EQ(-1.0, kCameraPosition.x);
		EXPECT_DOUBLE_EQ(-1.0, kCameraPosition.y);
		EXPECT_DOUBLE_EQ(-1.0, kCameraPosition.z);
	}

	{
		const cv::Matx33d kRotationMatrix(1.0, 0.0, 0.0
			, 0.0, 1.0, 0.0
			, 0.0, 0.0, 1.0);
		const cv::Matx31d kTranslation(0.0, 0.0, 0.0);
		const cv::Point3d kCameraPosition = CvUtil::computeCameraPosition(kRotationMatrix, kTranslation);
		EXPECT_DOUBLE_EQ(0.0, kCameraPosition.x);
		EXPECT_DOUBLE_EQ(0.0, kCameraPosition.y);
		EXPECT_DOUBLE_EQ(0.0, kCameraPosition.z);
	}

	{
		const cv::Matx33d kRotationMatrix(std::sqrt(0.5), std::sqrt(0.25), std::sqrt(0.25)
			, std::sqrt(0.25), std::sqrt(0.5), std::sqrt(0.25)
			, std::sqrt(0.25), std::sqrt(0.25), std::sqrt(0.5));
		const cv::Matx31d kTranslation(10.0, 10.0, 10.0);
		const cv::Point3d kCameraPosition = CvUtil::computeCameraPosition(kRotationMatrix, kTranslation);
		EXPECT_DOUBLE_EQ(-(10.0*std::sqrt(0.5) + 10.0*std::sqrt(0.25) + 10.0*std::sqrt(0.25)), kCameraPosition.x);
		EXPECT_DOUBLE_EQ(-(10.0*std::sqrt(0.5) + 10.0*std::sqrt(0.25) + 10.0*std::sqrt(0.25)), kCameraPosition.y);
		EXPECT_DOUBLE_EQ(-(10.0*std::sqrt(0.5) + 10.0*std::sqrt(0.25) + 10.0*std::sqrt(0.25)), kCameraPosition.z);
	}
}

TEST(CvUtilTest, computeCameraPosition2) {
	{
		const cv::Matx34d kExtrinsicParameter(1.0, 0.0, 0.0, 1.0
			, 0.0, 1.0, 0.0, 1.0
			, 0.0, 0.0, 1.0, 1.0);
		const cv::Point3d kCameraPosition = CvUtil::computeCameraPosition(kExtrinsicParameter);
		EXPECT_DOUBLE_EQ(-1.0, kCameraPosition.x);
		EXPECT_DOUBLE_EQ(-1.0, kCameraPosition.y);
		EXPECT_DOUBLE_EQ(-1.0, kCameraPosition.z);
	}

	{
		const cv::Matx34d kExtrinsicParameter(1.0, 0.0, 0.0, 0.0
			, 0.0, 1.0, 0.0, 0.0
			, 0.0, 0.0, 1.0, 0.0);
		const cv::Point3d kCameraPosition = CvUtil::computeCameraPosition(kExtrinsicParameter);
		EXPECT_DOUBLE_EQ(0.0, kCameraPosition.x);
		EXPECT_DOUBLE_EQ(0.0, kCameraPosition.y);
		EXPECT_DOUBLE_EQ(0.0, kCameraPosition.z);
	}

	{
		const cv::Matx34d kExtrinsicParameter(std::sqrt(0.5), std::sqrt(0.25), std::sqrt(0.25), 10.0
			, std::sqrt(0.25), std::sqrt(0.5), std::sqrt(0.25), 10.0
			, std::sqrt(0.25), std::sqrt(0.25), std::sqrt(0.5), 10.0);
		const cv::Point3d kCameraPosition = CvUtil::computeCameraPosition(kExtrinsicParameter);
		EXPECT_DOUBLE_EQ(-(10.0*std::sqrt(0.5) + 10.0*std::sqrt(0.25) + 10.0*std::sqrt(0.25)), kCameraPosition.x);
		EXPECT_DOUBLE_EQ(-(10.0*std::sqrt(0.5) + 10.0*std::sqrt(0.25) + 10.0*std::sqrt(0.25)), kCameraPosition.y);
		EXPECT_DOUBLE_EQ(-(10.0*std::sqrt(0.5) + 10.0*std::sqrt(0.25) + 10.0*std::sqrt(0.25)), kCameraPosition.z);
	}
}

TEST(CvUtilTest, computeProjectionPoint) {
	{
		const cv::Matx34d kProjectionMatrix(10.0, 10.0, 10.0, 10.0
			, 10.0, 10.0, 10.0, 10.0
			, 10.0, 10.0, 10.0, 10.0);
		const cv::Point3d kWorldPoint(10.0, 10.0, 10.0);
		const cv::Point2d kProjectionPoint = CvUtil::computeProjectionPoint(kProjectionMatrix, kWorldPoint);
		EXPECT_DOUBLE_EQ(1.0, kProjectionPoint.x);
		EXPECT_DOUBLE_EQ(1.0, kProjectionPoint.y);
	}
}

TEST(CvUtilTest, computeReprojectionError) {
	{
		const cv::Matx34d kProjectionMatrix(10.0, 10.0, 10.0, 10.0
			, 10.0, 10.0, 10.0, 10.0
			, 10.0, 10.0, 10.0, 10.0);
		const cv::Point3d kWorldPoint(10.0, 10.0, 10.0);
		const cv::Point2d kImagePoint(1.0, 1.0);
		const double kReprojectionError = CvUtil::computeReprojectionError(kImagePoint, kProjectionMatrix, kWorldPoint);
		EXPECT_DOUBLE_EQ(0.0, kReprojectionError);
	}
}

TEST(CvUtilTest, computeCameraParameter) {
	{
		const double kResidualThreshold = 10E-10;
		std::vector<cv::Point2d> image_points;
		const std::vector<cv::Point3d> kWorldPoints = { cv::Point3d(0.0, 0.0, 0.0)
			, cv::Point3d(1.0, 0.0, 0.0)
			, cv::Point3d(0.0, 1.0, 0.0)
			, cv::Point3d(2.0, 3.0, 5.0)
			, cv::Point3d(4.0, 1.0, 2.0)
			, cv::Point3d(0.0, 0.0, 1.0) };
		const cv::Matx33d kIntrinsicParameter(100.0, 0.0, 400.0
			, 0.0, 100.0, 200.0
			, 0.0, 0.0, 1.0);
		const cv::Matx34d kExtrinsicParameter(1.0, 0.0, 0.0, 1.0
			, 0.0, 1.0, 0.0, 2.0
			, 0.0, 0.0, 1.0, 3.0);
		cv::Matx34d projection_matrix = kIntrinsicParameter * kExtrinsicParameter;
		for (const auto& point : kWorldPoints) {
			const cv::Matx41d kWorldPoint(point.x, point.y, point.z, 1.0);
			const cv::Matx31d kImagePoint = projection_matrix * kWorldPoint;
			image_points.push_back(cv::Point2d(kImagePoint(0) / kImagePoint(2), kImagePoint(1) / kImagePoint(2)));
		}
		projection_matrix *= 1.0 / projection_matrix(2, 3);
		cv::Matx34d expected_projection_matrix = CvUtil::computeCameraParameter(image_points, kWorldPoints);
		expected_projection_matrix *= 1.0 / expected_projection_matrix(2, 3);
		EXPECT_NEAR(projection_matrix(0, 0), expected_projection_matrix(0, 0), kResidualThreshold);
		EXPECT_NEAR(projection_matrix(0, 1), expected_projection_matrix(0, 1), kResidualThreshold);
		EXPECT_NEAR(projection_matrix(0, 2), expected_projection_matrix(0, 2), kResidualThreshold);
		EXPECT_NEAR(projection_matrix(0, 3), expected_projection_matrix(0, 3), kResidualThreshold);
		EXPECT_NEAR(projection_matrix(1, 0), expected_projection_matrix(1, 0), kResidualThreshold);
		EXPECT_NEAR(projection_matrix(1, 1), expected_projection_matrix(1, 1), kResidualThreshold);
		EXPECT_NEAR(projection_matrix(1, 2), expected_projection_matrix(1, 2), kResidualThreshold);
		EXPECT_NEAR(projection_matrix(1, 3), expected_projection_matrix(1, 3), kResidualThreshold);
		EXPECT_NEAR(projection_matrix(2, 0), expected_projection_matrix(2, 0), kResidualThreshold);
		EXPECT_NEAR(projection_matrix(2, 1), expected_projection_matrix(2, 1), kResidualThreshold);
		EXPECT_NEAR(projection_matrix(2, 2), expected_projection_matrix(2, 2), kResidualThreshold);
		EXPECT_NEAR(projection_matrix(2, 3), expected_projection_matrix(2, 3), kResidualThreshold);
	}

	{
		const std::vector<cv::Point2d> kImagePoints = { cv::Point2d(0.0, 0.0), cv::Point2d(1.0, 0.0), cv::Point2d(2.0, 0.0), cv::Point2d(0.0, 1.0), cv::Point2d(0.0, 2.0) };
		const std::vector<cv::Point3d> kWorldPoints = { cv::Point3d(0.0, 0.0, 0.0), cv::Point3d(1.0, 0.0, 0.0), cv::Point3d(0.0, 2.0, 0.0), cv::Point3d(0.0, 0.0, 3.0) };
		testing::internal::CaptureStdout();
		const cv::Matx34d kProjectionMatrix = CvUtil::computeCameraParameter(kImagePoints, kWorldPoints);
		EXPECT_EQ(cv::Matx34d(), kProjectionMatrix);
		EXPECT_STREQ("[ERROR] The number of image points and world points are different.\n", testing::internal::GetCapturedStdout().c_str());
	}

	{
		const std::vector<cv::Point2d> kImagePoints = { cv::Point2d(0.0, 0.0), cv::Point2d(1.0, 0.0), cv::Point2d(2.0, 0.0), cv::Point2d(0.0, 1.0) };
		const std::vector<cv::Point3d> kWorldPoints = { cv::Point3d(0.0, 0.0, 0.0), cv::Point3d(1.0, 0.0, 0.0), cv::Point3d(0.0, 2.0, 0.0), cv::Point3d(0.0, 0.0, 3.0) };
		testing::internal::CaptureStdout();
		const cv::Matx34d kProjectionMatrix = CvUtil::computeCameraParameter(kImagePoints, kWorldPoints);
		EXPECT_EQ(cv::Matx34d(), kProjectionMatrix);
		EXPECT_STREQ("[ERROR] The number of points is not enough.\n", testing::internal::GetCapturedStdout().c_str());
	}
}

TEST(CvUtilTest, decomposeProjectionMatrix) {
	{
		const cv::Matx33d kIntrinsicParameter(100.0, 0.0, 400.0
			, 0.0, 100.0, 200.0
			, 0.0, 0.0, 1.0);
		const cv::Matx34d kExtrinsicParameter(1.0, 0.0, 0.0, 1.0
			, 0.0, 1.0, 0.0, 2.0
			, 0.0, 0.0, 1.0, 3.0);
		const cv::Matx34d kProjectionMatrix = kIntrinsicParameter * kExtrinsicParameter;
		cv::Matx33d intrinsic_parameter;
		cv::Matx33d rotation_matrix;
		cv::Matx31d translation_vector;
		CvUtil::decomposeProjectionMatrix(kProjectionMatrix, intrinsic_parameter, rotation_matrix, translation_vector);
		EXPECT_DOUBLE_EQ(kIntrinsicParameter(0, 0), intrinsic_parameter(0, 0));
		EXPECT_DOUBLE_EQ(kIntrinsicParameter(0, 1), intrinsic_parameter(0, 1));
		EXPECT_DOUBLE_EQ(kIntrinsicParameter(0, 2), intrinsic_parameter(0, 2));
		EXPECT_DOUBLE_EQ(kIntrinsicParameter(1, 0), intrinsic_parameter(1, 0));
		EXPECT_DOUBLE_EQ(kIntrinsicParameter(1, 1), intrinsic_parameter(1, 1));
		EXPECT_DOUBLE_EQ(kIntrinsicParameter(1, 2), intrinsic_parameter(1, 2));
		EXPECT_DOUBLE_EQ(kIntrinsicParameter(2, 0), intrinsic_parameter(2, 0));
		EXPECT_DOUBLE_EQ(kIntrinsicParameter(2, 1), intrinsic_parameter(2, 1));
		EXPECT_DOUBLE_EQ(kIntrinsicParameter(2, 2), intrinsic_parameter(2, 2));

		EXPECT_DOUBLE_EQ(kExtrinsicParameter(0, 0), rotation_matrix(0, 0));
		EXPECT_DOUBLE_EQ(kExtrinsicParameter(0, 1), rotation_matrix(0, 1));
		EXPECT_DOUBLE_EQ(kExtrinsicParameter(0, 2), rotation_matrix(0, 2));
		EXPECT_DOUBLE_EQ(kExtrinsicParameter(1, 0), rotation_matrix(1, 0));
		EXPECT_DOUBLE_EQ(kExtrinsicParameter(1, 1), rotation_matrix(1, 1));
		EXPECT_DOUBLE_EQ(kExtrinsicParameter(1, 2), rotation_matrix(1, 2));
		EXPECT_DOUBLE_EQ(kExtrinsicParameter(2, 0), rotation_matrix(2, 0));
		EXPECT_DOUBLE_EQ(kExtrinsicParameter(2, 1), rotation_matrix(2, 1));
		EXPECT_DOUBLE_EQ(kExtrinsicParameter(2, 2), rotation_matrix(2, 2));

		EXPECT_DOUBLE_EQ(kExtrinsicParameter(0, 3), translation_vector(0));
		EXPECT_DOUBLE_EQ(kExtrinsicParameter(1, 3), translation_vector(1));
		EXPECT_DOUBLE_EQ(kExtrinsicParameter(2, 3), translation_vector(2));
	}
}

TEST(CvUtilTest, triangulatePoints) {
	{
		const cv::Matx33d kIntrinsicParameter1(100.0, 0.0, 400.0
			, 0.0, 100.0, 200.0
			, 0.0, 0.0, 1.0);
		const cv::Matx34d kExtrinsicParameter1(1.0, 0.0, 0.0, 1.0
			, 0.0, 1.0, 0.0, 2.0
			, 0.0, 0.0, 1.0, 3.0);
		const cv::Matx33d kIntrinsicParameter2(200.0, 0.0, 500.0
			, 0.0, 200.0, 400.0
			, 0.0, 0.0, 1.0);
		const cv::Matx34d kExtrinsicParameter2(0.0, 1.0, 0.0, 5.0
			, 0.0, 0.0, 1.0, 3.0
			, 1.0, 0.0, 0.0, 5.0);
		const cv::Matx41d kWorldPoint(5.0, 4.0, 3.0, 1.0);
		const std::vector<cv::Matx34d> kProjectionMatrix = { kIntrinsicParameter1*kExtrinsicParameter1, kIntrinsicParameter2*kExtrinsicParameter2 };
		const cv::Matx31d kProjectionPoint1 = kIntrinsicParameter1 * kExtrinsicParameter1*kWorldPoint;
		const cv::Matx31d kProjectionPoint2 = kIntrinsicParameter2 * kExtrinsicParameter2*kWorldPoint;
		const std::vector<cv::Point2d> kProjectionPoints = { cv::Point2d(kProjectionPoint1(0) / kProjectionPoint1(2), kProjectionPoint1(1) / kProjectionPoint1(2))
															, cv::Point2d(kProjectionPoint2(0) / kProjectionPoint2(2), kProjectionPoint2(1) / kProjectionPoint2(2)) };
		const cv::Point3d kTriangulatedPoint = CvUtil::triangulatePoints(kProjectionPoints, kProjectionMatrix);
		EXPECT_DOUBLE_EQ(kWorldPoint(0), kTriangulatedPoint.x);
		EXPECT_DOUBLE_EQ(kWorldPoint(1), kTriangulatedPoint.y);
		EXPECT_DOUBLE_EQ(kWorldPoint(2), kTriangulatedPoint.z);
	}

	{
		const std::vector<cv::Matx34d> kProjectionMatrix = {};
		const std::vector<cv::Point2d> kProjectionPoints = {};
		const cv::Point3d kTriangulatedPoint = CvUtil::triangulatePoints(kProjectionPoints, kProjectionMatrix);
		EXPECT_DOUBLE_EQ(0.0, kTriangulatedPoint.x);
		EXPECT_DOUBLE_EQ(0.0, kTriangulatedPoint.y);
		EXPECT_DOUBLE_EQ(0.0, kTriangulatedPoint.z);
	}

	{
		const std::vector<cv::Matx34d> kProjectionMatrix = { cv::Matx34d::ones(), cv::Matx34d::zeros() };
		const std::vector<cv::Point2d> kProjectionPoints = { cv::Point2d(0.0, 0.0), cv::Point2d(1.0, 0.0), cv::Point2d(0.0, 1.0) };
		const cv::Point3d kTriangulatedPoint = CvUtil::triangulatePoints(kProjectionPoints, kProjectionMatrix);
		EXPECT_DOUBLE_EQ(0.0, kTriangulatedPoint.x);
		EXPECT_DOUBLE_EQ(0.0, kTriangulatedPoint.y);
		EXPECT_DOUBLE_EQ(0.0, kTriangulatedPoint.z);
	}
}
