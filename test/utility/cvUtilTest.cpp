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
		const std::vector<cv::Point2d> kImagePoint = { cv::Point2d(0.0, 0.0), cv::Point2d(1.0, 0.0), cv::Point2d(2.0, 0.0), cv::Point2d(0.0, 1.0), cv::Point2d(0.0, 2.0) };
		const std::vector<cv::Point3d> kWorldPoint = { cv::Point3d(0.0, 0.0, 0.0), cv::Point3d(1.0, 0.0, 0.0), cv::Point3d(0.0, 2.0, 0.0), cv::Point3d(0.0, 0.0, 3.0) };
		testing::internal::CaptureStdout();
		const cv::Matx34d kProjectionMatrix = CvUtil::computeCameraParameter(kImagePoint, kWorldPoint);
		EXPECT_EQ(cv::Matx34d(), kProjectionMatrix);
		EXPECT_STREQ("[ERROR] The number of image points and world points are different.\n", testing::internal::GetCapturedStdout().c_str());
	}

	{
		const std::vector<cv::Point2d> kImagePoint = { cv::Point2d(0.0, 0.0), cv::Point2d(1.0, 0.0), cv::Point2d(2.0, 0.0), cv::Point2d(0.0, 1.0) };
		const std::vector<cv::Point3d> kWorldPoint = { cv::Point3d(0.0, 0.0, 0.0), cv::Point3d(1.0, 0.0, 0.0), cv::Point3d(0.0, 2.0, 0.0), cv::Point3d(0.0, 0.0, 3.0) };
		testing::internal::CaptureStdout();
		const cv::Matx34d kProjectionMatrix = CvUtil::computeCameraParameter(kImagePoint, kWorldPoint);
		EXPECT_EQ(cv::Matx34d(), kProjectionMatrix);
		EXPECT_STREQ("[ERROR] The number of image points and world points are under 6.\n", testing::internal::GetCapturedStdout().c_str());
	}
}
