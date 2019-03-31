#define _USE_MATH_DEFINES
#include "mathUtil.hpp"
#include <cmath>
#include <algorithm>

double MathUtil::convertDegreeToRadian(double degree) {
	return degree * (M_PI / 180.0);
}

double MathUtil::convertRadianToDegree(double radian) {
	return radian * (180.0 / M_PI);
}

double MathUtil::clamp(double v, double low, double high) {
	return std::min(std::max(v, low), high);
}
