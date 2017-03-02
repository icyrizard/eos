#ifndef EOS_TEST_HELPER_HPP
#define EOS_TEST_HELPER_HPP

#include "eos/morphablemodel/MorphableModel.hpp"

eos::morphablemodel::MorphableModel loadTestModel() {
	return eos::morphablemodel::load_model("../share/sfm_shape_3448.bin");
}

#endif //EOS_TEST_HELPER_HPP
