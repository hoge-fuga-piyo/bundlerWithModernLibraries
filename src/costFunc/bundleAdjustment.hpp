#ifndef BUNDLE_ADJUSTMENT_HPP
#define BUNDLE_ADJUSTMENT_HPP

#include <vector>
#include "image.hpp"
#include "tracking.hpp"

class BundleAdjustment {
public:
	void runBundleAdjustment(std::vector<Image>& images, Tracking& tracking);
private:
	void extractCameraParams(const std::vector<Image>& kImages, const std::shared_ptr<double>& intrinsic_params, const std::shared_ptr<double>& extrinsic_params) const;
	void extractWorldPoints(const Tracking& kTracking, const std::shared_ptr<double>& world_points) const;
	void setOptimizationWorldPoints(Tracking& tracking, const std::shared_ptr<double>& world_points) const;
	void setOptimizationCameraParams(std::vector<Image>& images, const std::shared_ptr<double>& intrinsic_params, const std::shared_ptr<double>& extrinsic_params) const;
};

#endif