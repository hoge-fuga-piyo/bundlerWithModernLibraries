#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <limits>
#include "mathUtil.hpp"

TEST(MathUtilTest, convertDegreeToRadian) {
	const double kAccuracy = 10.0E-10;
	{
		const double kRadian = MathUtil::convertDegreeToRadian(180.0);
		EXPECT_DOUBLE_EQ(M_PI, kRadian);
	}

	{
		const double kRadian = MathUtil::convertDegreeToRadian(0.0);
		EXPECT_DOUBLE_EQ(0.0, kRadian);
	}

	{
		const double kRadian = MathUtil::convertDegreeToRadian(90.0);
		EXPECT_DOUBLE_EQ(M_PI / 2.0, kRadian);
	}

	{
		const double kRadian = MathUtil::convertDegreeToRadian(60.0);
		EXPECT_DOUBLE_EQ(M_PI / 3.0, kRadian);
	}

	{
		const double kRadian = MathUtil::convertDegreeToRadian(120.0);
		EXPECT_DOUBLE_EQ((M_PI / 3.0) * 2.0, kRadian);
	}
}

TEST(MathUtilTest, convertRadianToDegree) {
	const double kAccuracy = 10.0E-10;
	{
		const double kDegree = MathUtil::convertRadianToDegree(M_PI);
		EXPECT_DOUBLE_EQ(180.0, kDegree);
	}

	{
		const double kDegree = MathUtil::convertRadianToDegree(0.0);
		EXPECT_DOUBLE_EQ(0.0, kDegree);
	}

	{
		const double kDegree = MathUtil::convertRadianToDegree(M_PI / 2.0);
		EXPECT_DOUBLE_EQ(90.0, kDegree);
	}

	{
		const double kDegree = MathUtil::convertRadianToDegree(M_PI / 3.0);
		EXPECT_DOUBLE_EQ(60.0, kDegree);
	}

	{
		const double kDegree = MathUtil::convertRadianToDegree((M_PI / 3.0) * 2.0);
		EXPECT_DOUBLE_EQ(120.0, kDegree);
	}
}

TEST(MathUtilTest, clamp) {
	{
		const double kClamp = MathUtil::clamp(3.5, 4.0, 10.0);
		EXPECT_DOUBLE_EQ(4.0, kClamp);
	}

	{
		const double kClamp = MathUtil::clamp(5.0, 4.0, 10.0);
		EXPECT_DOUBLE_EQ(5.0, kClamp);
	}

	{
		const double kClamp = MathUtil::clamp(11.0, 4.0, 10.0);
		EXPECT_DOUBLE_EQ(10.0, kClamp);
	}

	{
		const double kClamp = MathUtil::clamp(-2.0, -4.0, 10.0);
		EXPECT_DOUBLE_EQ(-2.0, kClamp);
	}
}

TEST(MathUtilTest, combination) {
	{
		unsigned long long combination = MathUtil::combination(10, 2);
		EXPECT_EQ(45, combination);
	}

	{
		unsigned long long combination = MathUtil::combination(100, 10);
		EXPECT_EQ(17310309456440, combination);
	}

	{
		unsigned long long combination = MathUtil::combination(0, 2);
		EXPECT_EQ(0, combination);
	}

	{
		unsigned long long combination = MathUtil::combination(5, 6);
		EXPECT_EQ(0, combination);
	}

	{
		unsigned long long combination = MathUtil::combination(5, 0);
		EXPECT_EQ(1, combination);
	}
}