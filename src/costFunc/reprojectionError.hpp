#ifndef REPROJECTION_ERROR_HPP
#define REPROJECTION_ERROR_HPP

#include <ceres/ceres.h>
#include <ceres/rotation.h>

class ReprojectionError {
public:
	ReprojectionError(double observed_x, double observed_y) : observed_x_(observed_x), observed_y_(observed_y) {}

	template <typename T>
	bool operator()(const T* const intrinsic_param, const T* const extrinsic_param, const T* const world_point, T* residuals) const {
		computeReprojectionError(intrinsic_param, extrinsic_param, world_point, residuals);

		return true;
	}

	static ceres::CostFunction* Create(const double observed_x, const double observed_y) {
		return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 6, 3>(
			new ReprojectionError(observed_x, observed_y)));
	}

	static ceres::CostFunction* CreateForCameraParameter(const double observed_x, const double observed_y) {
		return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 6>(
			new ReprojectionError(observed_x, observed_y)));
	}

private:
	double observed_x_;
	double observed_y_;

	template <typename T>
	void computeReprojectionError(const T* const intrinsic_param, const T* const extrinsic_param, const T* const world_point, T* residuals) const {
		// World coordinate -> Camera coordinate
		T rotation_vec[3] = { extrinsic_param[0], extrinsic_param[1], extrinsic_param[2] };
		T camera_point[3];
		ceres::AngleAxisRotatePoint(rotation_vec, world_point, camera_point);

		camera_point[0] += extrinsic_param[3];
		camera_point[1] += extrinsic_param[4];
		camera_point[2] += extrinsic_param[5];

		// Camera coordinate -> Image coordinate
		T focal_length = intrinsic_param[0];
		T cx = intrinsic_param[1];
		T cy = intrinsic_param[2];

		T image_point[2];
		image_point[0] = (focal_length*camera_point[0] + cx * camera_point[2]) / camera_point[2];
		image_point[1] = (focal_length*camera_point[1] + cy * camera_point[2]) / camera_point[2];

		// Reprojection error
		residuals[0] = image_point[0] - T(observed_x_);
		residuals[1] = image_point[1] - T(observed_y_);
	}
};

#endif
