#ifndef HASH_KEY_HPP
#define HASH_KEY_HPP

#include <tuple>

namespace std {
	template<>
	class hash<std::tuple<int, int>>{
	public:
		size_t operator () (const std::tuple<int, int> &map) const {
			return std::get<0>(map)*std::get<1>(map);
		}
	};
}

#endif