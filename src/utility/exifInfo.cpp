#include "exifInfo.hpp"
#include <iostream>

const std::string ExifInfo::kFocalLength_ = "Exif.Photo.FocalLength";
const std::string ExifInfo::kFocalPlaneXResolution_ = "Exif.Photo.FocalPlaneXResolution";
const std::string ExifInfo::kFocalPlaneYResolution_ = "Exif.Photo.FocalPlaneYResolution";
const std::string ExifInfo::kFocalPlaneResolutionUnit_ = "Exif.Photo.FocalPlaneResolutionUnit";

ExifInfo::ExifInfo() {}

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

bool ExifInfo::hasFocalLengthInMm() const {
	return haveInfo(kFocalLength_);
}

double ExifInfo::getFocalLengthInMm() const {
	Exiv2::ExifData& exif_data = image_->exifData();
	const double kFocalLength = static_cast<double>(exif_data[kFocalLength_].toFloat());

	return kFocalLength;
}

bool ExifInfo::hasFocalLengthInPixel() const {
	if (!hasFocalLengthInMm() || !hasFocalPlaneXResolution() || !hasFocalPlaneYResolution() || !hasFocalPlaneResolutionUnit()) {
		return false;
	}

	return true;
}

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

bool ExifInfo::haveInfo(const std::string & kKey) const {
	try {
		Exiv2::ExifData& exif_data = image_->exifData();
		exif_data[kKey];
	}
	catch (Exiv2::Error& err) {
		return false;
	}

	return true;
}

bool ExifInfo::hasFocalPlaneXResolution() const {
	return haveInfo(kFocalPlaneXResolution_);
}

bool ExifInfo::hasFocalPlaneYResolution() const {
	return haveInfo(kFocalPlaneYResolution_);
}

bool ExifInfo::hasFocalPlaneResolutionUnit() const {
	return haveInfo(kFocalPlaneResolutionUnit_);
}
