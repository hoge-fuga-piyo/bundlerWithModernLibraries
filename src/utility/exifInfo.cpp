#include "exifInfo.hpp"
#include <iostream>

const std::string ExifInfo::kFocalLength_ = "Exif.Photo.FocalLength";
const std::string ExifInfo::kFocalPlaneXResolution_ = "Exif.Photo.FocalPlaneXResolution";
const std::string ExifInfo::kFocalPlaneYResolution_ = "Exif.Photo.FocalPlaneYResolution";
const std::string ExifInfo::kFocalPlaneResolutionUnit_ = "Exif.Photo.FocalPlaneResolutionUnit";

ExifInfo::ExifInfo() {}

bool ExifInfo::loadImage(const std::string & kImagePath) {
	try {
		std::unique_ptr<Exiv2::Image> image = Exiv2::ImageFactory::open(kImagePath);
		image->readMetadata();
		exif_data_ = image->exifData();
		
		if (exif_data_.empty()) {
			return false;
		}
	}
	catch (Exiv2::Error& err) {
		std::cout << err.what() << std::endl;
		return false;
	}

	return true;
}

bool ExifInfo::hasFocalLengthInMm() {
	return haveInfo(kFocalLength_);
}

double ExifInfo::getFocalLengthInMm() {
	const double kFocalLength = static_cast<double>(exif_data_[kFocalLength_].toFloat());

	return kFocalLength;
}

bool ExifInfo::hasFocalLengthInPixel() {
	if (!hasFocalLengthInMm() || !hasFocalPlaneXResolution() || !hasFocalPlaneYResolution() || !hasFocalPlaneResolutionUnit()) {
		return false;
	}

	return true;
}

double ExifInfo::getFocalLengthInPixel() {
	const double kFocalPlaneXResolution = static_cast<double>(exif_data_[kFocalPlaneXResolution_].toFloat());
	const double kFocalPlaneYResolution = static_cast<double>(exif_data_[kFocalPlaneYResolution_].toFloat());
	
	const double kFocalPlaneResolution = (kFocalPlaneXResolution + kFocalPlaneYResolution) / 2.0;
	const std::string kUnit = exif_data_[kFocalPlaneResolutionUnit_].toString();
	double scale = 1.0 / 25.4; // inch
	if (kUnit == "3") {	// cm
		scale = 1.0 / 10.0;
	} else if (kUnit == "4") { // mm
		scale = 1.0;
	}

	double pixel_per_mm = kFocalPlaneResolution * scale;
	const double kFocalLengthMm = static_cast<double>(exif_data_[kFocalLength_].toFloat());
	const double kFocalLengthPixel = pixel_per_mm * kFocalLengthMm;
	std::cout << pixel_per_mm << ", " << kFocalLengthMm << std::endl;

	return kFocalLengthPixel;
}

bool ExifInfo::haveInfo(const std::string & kKey) {
	try {
		exif_data_[kKey];
	}
	catch (Exiv2::Error& err) {
		return false;
	}

	return true;
}

bool ExifInfo::hasFocalPlaneXResolution() {
	return haveInfo(kFocalPlaneXResolution_);
}

bool ExifInfo::hasFocalPlaneYResolution() {
	return haveInfo(kFocalPlaneYResolution_);
}

bool ExifInfo::hasFocalPlaneResolutionUnit() {
	return haveInfo(kFocalPlaneResolutionUnit_);
}
