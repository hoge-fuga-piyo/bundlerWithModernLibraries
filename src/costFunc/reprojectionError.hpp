#ifndef REPROJECTION_ERROR_HPP
#define REPROJECTION_ERROR_HPP

#include <ceres/ceres.h>
#include <ceres/rotation.h>

class ReprojectionError {
public:
	//ReprojectionError(double observed_x, double observed_y) : observed_x_(observed_x), observed_y_(observed_y) {}

	ReprojectionError(double observed_x, double observed_y, double cx, double cy) : observed_x_(observed_x), observed_y_(observed_y) {
		principle_point[0] = cx;
		principle_point[1] = cy;
	}

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

	// Bundle adjustment for optimizing focal length, rotation, translation and world points.
	template <typename T>
	bool operator()(const T* const focal_length, const T* const extrinsic_param, const T* const world_point, T* residuals) const {
		T rotation_vec[3] = { extrinsic_param[0], extrinsic_param[1], extrinsic_param[2] };
		T camera_point[3];
		ceres::AngleAxisRotatePoint(rotation_vec, world_point, camera_point);

		camera_point[0] += extrinsic_param[3];
		camera_point[1] += extrinsic_param[4];
		camera_point[2] += extrinsic_param[5];

		// Camera coordinate -> Image coordinate
		T cx = T(principle_point[0]);
		T cy = T(principle_point[1]);

		T image_point[2];
		image_point[0] = (focal_length[0]*camera_point[0] + cx * camera_point[2]) / camera_point[2];
		image_point[1] = (focal_length[0]*camera_point[1] + cy * camera_point[2]) / camera_point[2];

		// Reprojection error
		residuals[0] = image_point[0] - T(observed_x_);
		residuals[1] = image_point[1] - T(observed_y_);

		return true;
	}

	//// Bundle adjustment for optimizing intrinsic/extrinsic parameter and world points
	//template <typename T>
	//bool operator()(const T* const intrinsic_param, const T* const extrinsic_param, const T* const world_point, T* residuals) const {
	//	// World coordinate -> Camera coordinate
	//	T rotation_vec[3] = { extrinsic_param[0], extrinsic_param[1], extrinsic_param[2] };
	//	T camera_point[3];
	//	ceres::AngleAxisRotatePoint(rotation_vec, world_point, camera_point);

	//	camera_point[0] += extrinsic_param[3];
	//	camera_point[1] += extrinsic_param[4];
	//	camera_point[2] += extrinsic_param[5];

	//	// Camera coordinate -> Image coordinate
	//	T focal_length = intrinsic_param[0];
	//	T cx = intrinsic_param[1];
	//	T cy = intrinsic_param[2];

	//	T image_point[2];
	//	image_point[0] = (focal_length*camera_point[0] + cx * camera_point[2]) / camera_point[2];
	//	image_point[1] = (focal_length*camera_point[1] + cy * camera_point[2]) / camera_point[2];

	//	// Reprojection error
	//	residuals[0] = image_point[0] - T(observed_x_);
	//	residuals[1] = image_point[1] - T(observed_y_);

	//	return true;
	//}

	// Bundle adjustment for optimizing intrinsic/extrinsic parameter, world points and radial distortion
	template <typename T>
	bool operator()(const T* const intrinsic_param, const T* const extrinsic_param, const T* const world_point, const T* const radial_coefficient, T* residuals) const {
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

		// Consider radian distortion
		T rho2 = ceres::pow(image_point[0]/focal_length, 2.0) + ceres::pow(image_point[1]/focal_length, 2.0);
		T alpha = radial_coefficient[0] * rho2 + radial_coefficient[1] * rho2*rho2;

		// Reprojection error
		//residuals[0] = alpha * image_point[0] - T(observed_x_);
		//residuals[1] = alpha * image_point[1] - T(observed_y_);
		residuals[0] = alpha * image_point[0] - T(observed_x_) + T(10.0) * (ceres::pow(radial_coefficient[0], 2.0) + ceres::pow(radial_coefficient[1], 2.0));
		residuals[1] = alpha * image_point[1] - T(observed_y_) + T(10.0) * (ceres::pow(radial_coefficient[0], 2.0) + ceres::pow(radial_coefficient[1], 2.0));

		return true;
	}

	// Bundle adjustment for optimizing world points
	template <typename T>
	bool operator()(const T* const world_point, T* residuals) const {
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

	//static ceres::CostFunction* Create(const double kObservedX, const double kObservedY) {
	//	return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 6, 3>(
	//		new ReprojectionError(kObservedX, kObservedY)));
	//}

	static ceres::CostFunction* Create(const double kObservedX, const double kObservedY, const double kCx, const double kCy) {
		return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 1, 6, 3>(
			new ReprojectionError(kObservedX, kObservedY, kCx, kCy)));
	}

	//static ceres::CostFunction* CreateWithDistortion(const double kObservedX, const double kObservedY) {
	//	return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 6, 3, 2>(
	//		new ReprojectionError(kObservedX, kObservedY)));
	//}

	static ceres::CostFunction* Create(const double kObservedX, const double kObservedY, const double* const kIntrinsicParam, const double* const kExtrinsicParam) {
		return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3>(
			new ReprojectionError(kObservedX, kObservedY, kIntrinsicParam, kExtrinsicParam)));
	}

private:
	double observed_x_;
	double observed_y_;
	double intrinsic_param_[3];
	double extrinsic_param_[6];
	double principle_point[2];
};

#endif
