#include "bundleAdjustment.hpp"
#include "reprojectionError.hpp"

BundleAdjustment::BundleAdjustment() : kThreadNum(8) {}

BundleAdjustment::BundleAdjustment(unsigned int thread_num) : kThreadNum(thread_num) {}

void BundleAdjustment::runBundleAdjustment(std::vector<Image>& images, Tracking& tracking) const {
	std::cout << "Before BA" << std::endl;
	for (const auto& image : images) {
		if (image.isRecoveredExtrinsicParameter()) {
			std::cout << image.getIntrinsicParameter() << std::endl;
		}
	}

	const size_t kImageNum = images.size();
	std::shared_ptr<double> intrinsic_params(new double[kImageNum * 3], std::default_delete<double[]>());	// focal length, cx, cy
	std::shared_ptr<double> extrinsic_params(new double[kImageNum * 6], std::default_delete<double[]>());	// angle axis rotation, translation
	extractCameraParams(images, intrinsic_params, extrinsic_params);

	std::shared_ptr<double> radial_distortion(new double[kImageNum * 2], std::default_delete<double[]>());
	extractRadialDistortion(images, radial_distortion);

	const size_t kPointNum = tracking.getTrackingNum();
	std::shared_ptr<double> world_points(new double[kPointNum * 3], std::default_delete<double[]>());
	extractWorldPoints(tracking, world_points);

	double* intrinsic_params_pt = intrinsic_params.get();
	double* extrinsic_params_pt = extrinsic_params.get();
	double* radial_distortion_pt = radial_distortion.get();
	double* world_points_pt = world_points.get();
	ceres::Problem problem;
	for (int i = 0; i < kPointNum; i++) {
		if (!tracking.isRecoveredTriangulatedPoint(i)) {
			continue;
		}
		for (int j = 0; j < kImageNum; j++) {
			if (!images[j].isRecoveredExtrinsicParameter()) {
				continue;
			}

			const std::vector<cv::KeyPoint>& kKeypoints = images[j].getKeypoints();
			const int kKeypointIndex = tracking.getTrackedKeypointIndex(i, j);
			if (kKeypointIndex < 0) {
				continue;
			}
			const cv::KeyPoint kKeypoint = kKeypoints[kKeypointIndex];

			//ceres::CostFunction* cost_function = ReprojectionError::Create(kKeypoint.pt.x, kKeypoint.pt.y);
			ceres::CostFunction* cost_function = ReprojectionError::Create(kKeypoint.pt.x, kKeypoint.pt.y, intrinsic_params_pt[j * 3 + 1], intrinsic_params_pt[j * 3 + 2]);
			//ceres::CostFunction* cost_function = ReprojectionError::CreateWithDistortion(kKeypoint.pt.x, kKeypoint.pt.y);
			ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
			//ceres::LossFunction* loss_function = NULL;

			problem.AddResidualBlock(cost_function, loss_function, intrinsic_params_pt + (j * 3), extrinsic_params_pt + (j * 6), world_points_pt + (i * 3));
			//problem.AddResidualBlock(cost_function, loss_function, intrinsic_params_pt + (j * 3), extrinsic_params_pt + (j * 6), world_points_pt + (i * 3), radial_distortion_pt + (j * 2));
		}
	}
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
	options.minimizer_progress_to_stdout = true;
	options.max_num_iterations = 100000;
	options.num_threads = kThreadNum;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << std::endl;

	setOptimizationWorldPoints(tracking, world_points);
	setOptimizationCameraParams(images, intrinsic_params, extrinsic_params);

	std::cout << "After BA" << std::endl;
	for (const auto& image : images) {
		if (image.isRecoveredExtrinsicParameter()) {
			std::cout << image.getIntrinsicParameter() << std::endl;
		}
	}
	std::cout << "end" << std::endl;
}

/**
 * @brief Run bundle adjustment
 * @param[in,out] images Image information
 * @param[in,out] tracking The information of world points related image points in each image.
 * @param[in] optimized_indexes The target of bundle adjustment image. These intrinsic/extrinsic parameters and observed world points are optimized.
 */
void BundleAdjustment::runBundleAdjustment(std::vector<Image>& images, Tracking & tracking, const std::vector<int>& kOptimizationImageIndexes) const {
	std::cout << "Before BA" << std::endl;
	for (const auto& image : images) {
		if (image.isRecoveredExtrinsicParameter()) {
			std::cout << image.getIntrinsicParameter() << std::endl;
		}
	}
	std::cout << "end" << std::endl;

	const size_t kImageNum = images.size();
	std::shared_ptr<double> intrinsic_params(new double[kImageNum * 3], std::default_delete<double[]>());	// focal length, cx, cy
	std::shared_ptr<double> extrinsic_params(new double[kImageNum * 6], std::default_delete<double[]>());	// angle axis rotation, translation
	extractCameraParams(images, intrinsic_params, extrinsic_params);

	const size_t kPointNum = tracking.getTrackingNum();
	std::shared_ptr<double> world_points(new double[kPointNum * 3], std::default_delete<double[]>());
	extractWorldPoints(tracking, world_points);

	double* intrinsic_params_pt = intrinsic_params.get();
	double* extrinsic_params_pt = extrinsic_params.get();
	double* world_points_pt = world_points.get();

	std::unordered_map<int, bool> optimization_index_map;
	for (const int kIndex : kOptimizationImageIndexes) {
		optimization_index_map[kIndex] = true;
	}

	int count = 0;
	ceres::Problem problem;
	for (int i = 0; i < kPointNum; i++) {
		if (!tracking.isRecoveredTriangulatedPoint(i)) {
			continue;
		}

		bool isOptimizeTarget = isWorldPointObsevedOptimizationImages(tracking, i, kOptimizationImageIndexes);
		if (!isOptimizeTarget) {
			continue;
		}

		count++;

		for (int j = 0; j < kImageNum; j++) {
			if (!images[j].isRecoveredExtrinsicParameter()) {
				continue;
			}

			const std::vector<cv::KeyPoint>& kKeypoints = images[j].getKeypoints();
			const int kKeypointIndex = tracking.getTrackedKeypointIndex(i, j);
			if (kKeypointIndex < 0) {
				continue;
			}
			const cv::KeyPoint kKeypoint = kKeypoints[kKeypointIndex];

			ceres::CostFunction* cost_function;
			ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
			if (optimization_index_map.find(j) == optimization_index_map.end()) {
				cost_function = ReprojectionError::Create(kKeypoint.pt.x, kKeypoint.pt.y, intrinsic_params_pt + (j * 3), extrinsic_params_pt + (j * 6));
				problem.AddResidualBlock(cost_function, loss_function, world_points_pt + (i * 3));
			} else {
				cost_function = ReprojectionError::Create(kKeypoint.pt.x, kKeypoint.pt.y, intrinsic_params_pt[j * 3 + 1], intrinsic_params_pt[j * 3 + 2]);
				problem.AddResidualBlock(cost_function, loss_function, intrinsic_params_pt + (j * 3), extrinsic_params_pt + (j * 6), world_points_pt + (i * 3));
			}
		}
	}
	std::cout << "optimize world point: " << count << std::endl;
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
	options.minimizer_progress_to_stdout = true;
	options.max_num_iterations = 10000;
	options.num_threads = kThreadNum;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << std::endl;

	setOptimizationWorldPoints(tracking, world_points);
	setOptimizationCameraParams(images, intrinsic_params, extrinsic_params);

	std::cout << "After BA" << std::endl;
	for (const auto& image : images) {
		if (image.isRecoveredExtrinsicParameter()) {
			std::cout << image.getIntrinsicParameter() << std::endl;
		}
	}
	std::cout << "end" << std::endl;
}

void BundleAdjustment::extractCameraParams(const std::vector<Image>& kImages, const std::shared_ptr<double>& intrinsic_params, const std::shared_ptr<double>& extrinsic_params) const {
	double* intrinsic_params_pt = intrinsic_params.get();
	double* extrinsic_params_pt = extrinsic_params.get();
	const size_t kImageNum = kImages.size();
	for (size_t i = 0; i < kImageNum; i++) {
		if (!kImages[i].isRecoveredExtrinsicParameter()) {
			continue;
		}
		cv::Matx33d intrinsic_param = kImages[i].getIntrinsicParameter();
		intrinsic_params_pt[i * 3 + 0] = (intrinsic_param(0, 0) + intrinsic_param(1, 1)) / 2.0;
		intrinsic_params_pt[i * 3 + 1] = intrinsic_param(0, 2);
		intrinsic_params_pt[i * 3 + 2] = intrinsic_param(1, 2);

		cv::Vec3d rotation_vec = kImages[i].getRotationAngleAxis();
		extrinsic_params_pt[i * 6 + 0] = rotation_vec(0);
		extrinsic_params_pt[i * 6 + 1] = rotation_vec(1);
		extrinsic_params_pt[i * 6 + 2] = rotation_vec(2);
		cv::Matx31d translation = kImages[i].getTranslation();
		extrinsic_params_pt[i * 6 + 3] = translation(0, 0);
		extrinsic_params_pt[i * 6 + 4] = translation(1, 0);
		extrinsic_params_pt[i * 6 + 5] = translation(2, 0);
	}
}

void BundleAdjustment::extractRadialDistortion(const std::vector<Image>& kImages, const std::shared_ptr<double>& radial_distortion) const {
	double* radial_distortion_pt = radial_distortion.get();
	const size_t kImageNum = kImages.size();
	for (size_t i = 0; i < kImageNum; i++) {
		if (!kImages[i].isRecoveredExtrinsicParameter()) {
			continue;
		}
		std::array<double, 2> radial_distortion = kImages[i].getRadialDistortion();
		radial_distortion_pt[i * 2 + 0] = radial_distortion[0];
		radial_distortion_pt[i * 2 + 1] = radial_distortion[1];
	}
}

void BundleAdjustment::extractWorldPoints(const Tracking & kTracking, const std::shared_ptr<double>& world_points) const {
	double* world_points_pt = world_points.get();
	const int kPointNum = static_cast<int>(kTracking.getTrackingNum());
	for (int i = 0; i < kPointNum; i++) {
		if (!kTracking.isRecoveredTriangulatedPoint(i)) {
			continue;
		}
		const cv::Point3d& kPoint = kTracking.getTriangulatedPoint(i);
		world_points_pt[i * 3 + 0] = kPoint.x;
		world_points_pt[i * 3 + 1] = kPoint.y;
		world_points_pt[i * 3 + 2] = kPoint.z;
	}
}

void BundleAdjustment::setOptimizationWorldPoints(Tracking & tracking, const std::shared_ptr<double>& world_points) const {
	double* world_points_pt = world_points.get();
	const int kPointNum = static_cast<int>(tracking.getTrackingNum());
	for (int i = 0; i < kPointNum; i++) {
		if (!tracking.isRecoveredTriangulatedPoint(i)) {
			continue;
		}
		tracking.setTriangulatedPoint(i, world_points_pt[i * 3 + 0], world_points_pt[i * 3 + 1], world_points_pt[i * 3 + 2]);
	}
}

void BundleAdjustment::setOptimizationCameraParams(std::vector<Image>& images, const std::shared_ptr<double>& intrinsic_params, const std::shared_ptr<double>& extrinsic_params) const {
	double* intrinsic_params_pt = intrinsic_params.get();
	double* extrinsic_params_pt = extrinsic_params.get();
	const size_t kImageNum = images.size();
	for (size_t i = 0; i < kImageNum; i++) {
		if (!images[i].isRecoveredExtrinsicParameter()) {
			continue;
		}
		images[i].setFocalLength(intrinsic_params_pt[i * 3 + 0]);
		images[i].setPrincipalPoint(intrinsic_params_pt[i * 3 + 1], intrinsic_params_pt[i * 3 + 2]);

		std::cout << "optimized intrinsic: " << std::endl;
		std::cout << images[i].getIntrinsicParameter() << std::endl;

		cv::Vec3d rotation_vec(extrinsic_params_pt[i * 6 + 0], extrinsic_params_pt[i * 6 + 1], extrinsic_params_pt[i * 6 + 2]);
		cv::Matx33d rotation_mat;
		cv::Rodrigues(rotation_vec, rotation_mat);
		cv::Matx31d translation_vec(extrinsic_params_pt[i * 6 + 3], extrinsic_params_pt[i * 6 + 4], extrinsic_params_pt[i * 6 + 5]);
		images[i].setExtrinsicParameter(rotation_mat, translation_vec);
	}
}

bool BundleAdjustment::isWorldPointObsevedOptimizationImages(const Tracking & tracking, int track_index, const std::vector<int>& kOptimizationIndexes) const {
	for (size_t i = 0; i < kOptimizationIndexes.size(); i++) {
		const int kKeypointIndex = tracking.getTrackedKeypointIndex(track_index, kOptimizationIndexes[i]);
		if (kKeypointIndex >= 0) {
			return true;
		}
	}
	return false;
}
