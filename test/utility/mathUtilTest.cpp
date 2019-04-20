#include <gtest/gtest.h>
#include <limits>
#include "mathUtil.hpp"

TEST(MathUtilTest, clamp) {
	{
		double clamp = MathUtil::clamp(3.5, 4.0, 10.0);
		EXPECT_TRUE(std::fabs(clamp - 4.0) < std::numeric_limits<double>::epsilon());
	}

	{
		double clamp = MathUtil::clamp(5.0, 4.0, 10.0);
		EXPECT_TRUE(std::fabs(clamp - 5.0) < std::numeric_limits<double>::epsilon());
	}

	{
		double clamp = MathUtil::clamp(11.0, 4.0, 10.0);
		EXPECT_TRUE(std::fabs(clamp - 10.0) < std::numeric_limits<double>::epsilon());
	}

	{
		double clamp = MathUtil::clamp(-2.0, -4.0, 10.0);
		EXPECT_TRUE(std::fabs(clamp + 2.0) < std::numeric_limits<double>::epsilon());
	}
}