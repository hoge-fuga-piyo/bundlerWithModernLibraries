#ifndef REPROJECTION_ERROR_HPP
#define REPROJECTION_ERROR_HPP

#include <ceres/ceres.h>
#include <ceres/rotation.h>

class ReprojectionError {
public:
	ReprojectionError(double observed_x, double observed_y) : observed_x_(observed_x), observed_y_(observed_y) {}
	ReprojectionError(double observed_x, double observed_y, const double* const kIntrinsicParam, const double* const kExtrinsicParam) : observed_x_(observed_x), observed_y_(observed_y) {
		intrinsic_param_[0] = kIntrinsicParam[0];
		intrinsic_param_[1] = kIntrinsicParam[1];
		intrinsic_param_[2] = kIntrinsicParam[2];
		extrinsic_param_[0] = kExtrinsicParam[0];
		extrinsic_param_[1] = kExtrinsicParam[1];
		extrinsic_param_[2] = kExtrinsicParam[2];
		extrinsic_param_[3] = kExtrinsicParam[3];
		extrinsic_param_[4] = kExtrinsicParam[4];
		extrinsic_param_[5] = kExtrinsicParam[5];
	}

	template <typename T>
	bool operator()(const T* const intrinsic_param, const T* const extrinsic_param, const T* const world_point, T* residuals) const {
		computeReprojectionError(intrinsic_param, extrinsic_param, world_point, residuals);
		return true;
	}

	template <typename T>
	bool operator()(const T* const world_point, T* residuals) const {
		//computeReprojectionError((T*)intrinsic_param_, (T*)extrinsic_param_, world_point, residuals);
		// World coordinate -> Camera coordinate
		T rotation_vec[3] = { T(extrinsic_param_[0]), T(extrinsic_param_[1]), T(extrinsic_param_[2]) };
		T camera_point[3];
		ceres::AngleAxisRotatePoint(rotation_vec, world_point, camera_point);

		camera_point[0] += T(extrinsic_param_[3]);
		camera_point[1] += T(extrinsic_param_[4]);
		camera_point[2] += T(extrinsic_param_[5]);

		// Camera coordinate -> Image coordinate
		T focal_length = T(intrinsic_param_[0]);
		T cx = T(intrinsic_param_[1]);
		T cy = T(intrinsic_param_[2]);

		T image_point[2];
		image_point[0] = (focal_length*camera_point[0] + cx * camera_point[2]) / camera_point[2];
		image_point[1] = (focal_length*camera_point[1] + cy * camera_point[2]) / camera_point[2];

		// Reprojection error
		residuals[0] = image_point[0] - T(observed_x_);
		residuals[1] = image_point[1] - T(observed_y_);
		return true;
	}

	static ceres::CostFunction* Create(const double kObservedX, const double kObservedY) {
		return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 6, 3>(
			new ReprojectionError(kObservedX, kObservedY)));
	}

	static ceres::CostFunction* Create(const double kObservedX, const double kObservedY, const double* const kIntrinsicParam, const double* const kExtrinsicParam) {
		return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3>(
			new ReprojectionError(kObservedX, kObservedY, kIntrinsicParam, kExtrinsicParam)));
	}

private:
	double observed_x_;
	double observed_y_;
	double intrinsic_param_[3];
	double extrinsic_param_[6];

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
