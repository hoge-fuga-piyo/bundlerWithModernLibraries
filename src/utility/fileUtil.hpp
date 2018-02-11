#ifndef FILE_UTIL_HPP
#define FILE_UTIL_HPP

#include <iostream>
#include <vector>
#include <array>
#include <filesystem>

class FileUtil {
public:

	static std::tuple<std::vector<std::experimental::filesystem::path>
		, std::vector<std::experimental::filesystem::path>>
		readFilesAndDirs(const std::string& dir_path);
	static std::vector<std::experimental::filesystem::path> readFiles(const std::string& kDirPath);
private:

};

#endif