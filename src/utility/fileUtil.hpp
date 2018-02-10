#ifndef FILE_UTIL_HPP
#define FILE_UTIL_HPP

#include <iostream>
#include <vector>
#include <array>
#include <filesystem>

class FileUtil {
public:
	static std::vector<std::experimental::filesystem::path> readFileList(const std::string& dir_path);
private:

};

#endif