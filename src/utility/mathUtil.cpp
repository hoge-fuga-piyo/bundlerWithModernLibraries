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

unsigned long long MathUtil::combination(unsigned int all_num, unsigned int select_num) {
	if (select_num == 0) {
		return 1;
	}
	if (all_num < select_num) {
		return 0;
	}

	unsigned int num = all_num;
	unsigned long long combination = 1;
	for (unsigned int i = 1; i <= select_num; i++) {
		combination *= num;
		combination /= i;
		num--;
	}

	return combination;
}
