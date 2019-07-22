#ifndef EXIF_INFO_HPP
#define EXIF_INFO_HPP

#include <exiv2/exiv2.hpp>

class ExifInfo {
public:
	ExifInfo();
	bool loadImage(const std::string& kImagePath);
	bool hasFocalLengthInMm() const;
	double getFocalLengthInMm() const;
	bool hasFocalLengthInPixel() const;
	double getFocalLengthInPixel() const;

private:
	std::unique_ptr<Exiv2::Image> image_;
	
	static const std::string kFocalLength_;
	static const std::string kFocalPlaneXResolution_;
	static const std::string kFocalPlaneYResolution_;
	static const std::string kFocalPlaneResolutionUnit_;

	bool haveInfo(const std::string& kKey) const;
	bool hasFocalPlaneXResolution() const;
	bool hasFocalPlaneYResolution() const;
	bool hasFocalPlaneResolutionUnit() const;
};

#endif