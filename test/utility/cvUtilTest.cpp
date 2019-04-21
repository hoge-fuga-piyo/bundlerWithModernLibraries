#include <gtest/gtest.h>
#include <limits>
#include "cvUtil.hpp"

TEST(CvUtilTest, convertImagePointToCameraVector) {
	const cv::Point2d kImagePoint(5.0, 10.0);
	const cv::Matx33d kIntrinsicParameter(100.0, 0.0, 500.0
		, 0.0, 100.0, 400.0
		, 0.0, 0.0, 1.0);

	{
		const cv::Point3d kCameraVector = CvUtil::convertImagePointToCameraVector(kImagePoint, kIntrinsicParameter);
		std::cout << kCameraVector << std::endl;
		EXPECT_DOUBLE_EQ(-4.95, kCameraVector.x);
		EXPECT_DOUBLE_EQ(-3.9, kCameraVector.y);
		EXPECT_DOUBLE_EQ(1.0, kCameraVector.z);
	}
}
