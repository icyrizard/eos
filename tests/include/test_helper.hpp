#ifndef EOS_TEST_HELPER_HPP
#define EOS_TEST_HELPER_HPP


#include "eos/morphablemodel/MorphableModel.hpp"

std::string getTestPrefixPath() {
	return (std::string) std::getenv("TEST_PREFIX_PATH");
}

eos::morphablemodel::MorphableModel loadTestModel() {
	return eos::morphablemodel::load_model((std::string)std::getenv("TEST_PREFIX_PATH") + "/share/sfm_shape_3448.bin");
}

#endif //EOS_TEST_HELPER_HPP
