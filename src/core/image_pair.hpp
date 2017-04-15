#include <iostream>
#include <vector>
#include <array>
#include <opencv2/opencv.hpp>

class ImagePair{
public:
private:
    std::array<int, 2> index;       //! index of two images
    std::vector<cv::DMatch> match;
};