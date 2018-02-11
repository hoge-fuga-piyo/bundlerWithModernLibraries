#include "fileUtil.hpp"

/**
 * @brief Read file and directory paths in the target directory.
 * @param[in] kDirPath target directory
 * @return File paths and directory paths
 */
std::tuple<std::vector<std::experimental::filesystem::path>
	, std::vector<std::experimental::filesystem::path>>
	FileUtil::readFilesAndDirs(const std::string & kDirPath) {

	namespace fs = std::experimental::filesystem;
	
	std::vector<fs::path> file_paths;
	std::vector<fs::path> dir_paths;
	const fs::path kPath(kDirPath);
	std::for_each(fs::directory_iterator(kPath)
		, fs::directory_iterator()
		, [&file_paths, &dir_paths](const fs::path p) {
		if (fs::is_regular_file(p)) {
			file_paths.push_back(p);

		}
		else if(fs::is_directory(p)){
			dir_paths.push_back(p);
		}
	});

	return std::move(std::tuple<std::vector<fs::path>, std::vector<fs::path>>(file_paths, dir_paths));
}

/**
 * @brief Read file paths in the target directory.
 * @param[in] kDirPath target directory
 * @return File paths
 */
std::vector<std::experimental::filesystem::path> FileUtil::readFiles(const std::string & kDirPath) {
	const auto files_and_dirs = readFilesAndDirs(kDirPath);

	return std::move(std::get<0>(files_and_dirs));
}
