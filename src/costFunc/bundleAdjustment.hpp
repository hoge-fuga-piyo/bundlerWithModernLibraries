#ifndef BUNDLE_ADJUSTMENT_HPP
#define BUNDLE_ADJUSTMENT_HPP

#include <vector>
#include "image.hpp"
#include "tracking.hpp"

class BundleAdjustment {
public:
	BundleAdjustment();
	BundleAdjustment(unsigned int thread_num);
	void runBundleAdjustment(std::vector<Image>& images, Tracking& tracking) const;
	void runBundleAdjustment(std::vector<Image>& images, Tracking& tracking, const std::vector<int>& kOptimizationImageIndexes) const;
private:
	const unsigned int kThreadNum;

	void extractCameraParams(const std::vector<Image>& kImages, const std::shared_ptr<double>& intrinsic_params, const std::shared_ptr<double>& extrinsic_params) const;
	void extractRadialDistortion(const std::vector<Image>& kImages, const std::shared_ptr<double>& radial_distortion) const;
	void extractWorldPoints(const Tracking& kTracking, const std::shared_ptr<double>& world_points) const;
	void setOptimizationWorldPoints(Tracking& tracking, const std::shared_ptr<double>& world_points) const;
	void setOptimizationCameraParams(std::vector<Image>& images, const std::shared_ptr<double>& intrinsic_params, const std::shared_ptr<double>& extrinsic_params) const;
	bool isWorldPointObsevedOptimizationImages(const Tracking& tracking, int track_index, const std::vector<int>& kOptimizationIndexes) const;
};

#endif