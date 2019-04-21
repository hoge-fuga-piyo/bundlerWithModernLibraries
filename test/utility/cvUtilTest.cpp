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
