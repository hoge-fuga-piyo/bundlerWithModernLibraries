#include "exifInfo.hpp"
#include <iostream>

const std::string ExifInfo::kFocalLength_ = "Exif.Photo.FocalLength";
const std::string ExifInfo::kFocalPlaneXResolution_ = "Exif.Photo.FocalPlaneXResolution";
const std::string ExifInfo::kFocalPlaneYResolution_ = "Exif.Photo.FocalPlaneYResolution";
const std::string ExifInfo::kFocalPlaneResolutionUnit_ = "Exif.Photo.FocalPlaneResolutionUnit";

ExifInfo::ExifInfo() {}

/**
 * @brief load exif from a image
 * @param[in] kImagePath image file path
 * @return return true if exif is loaded success
 */
bool ExifInfo::loadImage(const std::string & kImagePath) {
	try {
		image_ = Exiv2::ImageFactory::open(kImagePath);
		image_->readMetadata();
		Exiv2::ExifData& exif_data = image_->exifData();
		
		if (exif_data.empty()) {
			std::cout << "has not exif" << std::endl;
			return false;
		}
		std::cout << "has exif" << std::endl;
	}
	catch (Exiv2::Error& err) {
		std::cout << err.what() << std::endl;
		return false;
	}

	return true;
}

/**
 * @brief check that exif has focal length expressed in mm
 * @return return true if exif has focal length in mm
 */
bool ExifInfo::hasFocalLengthInMm() const {
	return haveInfo(kFocalLength_);
}

/**
 * @brief get focal length expressed in mm in exif
 * @return focal length expressed in mm
 */
double ExifInfo::getFocalLengthInMm() const {
	Exiv2::ExifData& exif_data = image_->exifData();
	const double kFocalLength = static_cast<double>(exif_data[kFocalLength_].toFloat());

	return kFocalLength;
}

/**
 * @brief check that focal length expressed in pixel can be computed from exif
 * @return return true if focal length expressed in pixel can be computed
 */
bool ExifInfo::hasFocalLengthInPixel() const {
	if (!hasFocalLengthInMm() || !hasFocalPlaneXResolution() || !hasFocalPlaneYResolution() || !hasFocalPlaneResolutionUnit()) {
		return false;
	}

	return true;
}

/**
 * @brief compute focal length expressed in pixel from exif
 * @return focal length expressed in pixel
 */
double ExifInfo::getFocalLengthInPixel() const {
	Exiv2::ExifData& exif_data = image_->exifData();
	const double kFocalPlaneXResolution = static_cast<double>(exif_data[kFocalPlaneXResolution_].toFloat());
	const double kFocalPlaneYResolution = static_cast<double>(exif_data[kFocalPlaneYResolution_].toFloat());
	
	const double kFocalPlaneResolution = (kFocalPlaneXResolution + kFocalPlaneYResolution) / 2.0;
	const std::string kUnit = exif_data[kFocalPlaneResolutionUnit_].toString();
	double scale = 1.0 / 25.4; // inch
	if (kUnit == "3") {	// cm
		scale = 1.0 / 10.0;
	} else if (kUnit == "4") { // mm
		scale = 1.0;
	}

	double pixel_per_mm = kFocalPlaneResolution * scale;
	const double kFocalLengthMm = static_cast<double>(exif_data[kFocalLength_].toFloat());
	const double kFocalLengthPixel = pixel_per_mm * kFocalLengthMm;

	return kFocalLengthPixel;
}

/**
 * @brief check that exif has kKey
 * @return return true if exif has kKey
 */
bool ExifInfo::haveInfo(const std::string & kKey) const {
	try {
		Exiv2::ExifData& exif_data = image_->exifData();
		exif_data[kKey];
	}
	catch (Exiv2::Error /*err*/) {
		return false;
	}

	return true;
}

/**
 * @brief check that exif has FocalPlaneXResolution
 * @return return if exif has FocalPlaneXResolution
 */
bool ExifInfo::hasFocalPlaneXResolution() const {
	return haveInfo(kFocalPlaneXResolution_);
}

/**
 * @brief check that exif has FocalPlaneYResolution
 * @return return if exif has FocalPlaneYResolution
 */
bool ExifInfo::hasFocalPlaneYResolution() const {
	return haveInfo(kFocalPlaneYResolution_);
}

/**
 * @brief check that exif has FocalPlaneResolutionUnit
 * @return return if exif has FocalPlaneResolutionUnit
 */
bool ExifInfo::hasFocalPlaneResolutionUnit() const {
	return haveInfo(kFocalPlaneResolutionUnit_);
}
