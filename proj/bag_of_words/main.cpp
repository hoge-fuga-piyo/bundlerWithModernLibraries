#include <iostream>
#include <opencv2/opencv.hpp>
#include "core.hpp"
#include "image_pair.hpp"
#include "fileUtil.hpp"

int main(){
	std::vector<std::experimental::filesystem::path> paths = FileUtil::readFiles("./");
	for (const auto& path : paths) {
		std::cout << path.filename()<<std::endl;
	}
	
	return 0;
}