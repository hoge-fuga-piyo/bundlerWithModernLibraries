#ifndef EXIF_INFO_HPP
#define EXIF_INFO_HPP

#include <exiv2/exiv2.hpp>

class ExifInfo {
public:
	ExifInfo();
	bool loadImage(const std::string& kImagePath);
	bool hasFocalLengthInMm();
	double getFocalLengthInMm();
	bool hasFocalLengthInPixel();
	double getFocalLengthInPixel();

private:
	Exiv2::ExifData exif_data_;
	
	static const std::string kFocalLength_;
	static const std::string kFocalPlaneXResolution_;
	static const std::string kFocalPlaneYResolution_;
	static const std::string kFocalPlaneResolutionUnit_;

	bool haveInfo(const std::string& kKey);
	bool hasFocalPlaneXResolution();
	bool hasFocalPlaneYResolution();
	bool hasFocalPlaneResolutionUnit();
};

#endif