#include "fileUtil.hpp"

std::vector<std::experimental::filesystem::path> FileUtil::readFileList(const std::string & kDirPath) {
	namespace filesystem = std::experimental::filesystem;

	std::vector<filesystem::path> file_paths;
	const filesystem::path kPath(kDirPath);
	std::for_each(filesystem::directory_iterator(kPath)
		, filesystem::directory_iterator()
		, [&file_paths](const filesystem::path p) {
		if (filesystem::is_regular_file(p)) {
			std::cout << "file: " << p.filename() << std::endl;
		}
		else if (filesystem::is_directory(p)) {
			std::cout << "dir: " << p.filename() << std::endl;
		}
		file_paths.push_back(p.filename());
	});

	return file_paths;
}
