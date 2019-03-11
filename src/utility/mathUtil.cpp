#define _USE_MATH_DEFINES
#include "mathUtil.hpp"
#include <cmath>

double MathUtil::convertDegreeToRadian(double degree) {
	return degree * (M_PI / 180.0);
}
