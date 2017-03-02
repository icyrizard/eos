// Example file, show to use Catch with multiple files through one main_tests file.
#pragma once

#ifndef EOS_LANDMARK_TEST_HPP_
#define EOS_LANDMARK_TEST_HPP_

#include "catch.hpp"


TEST_CASE("Test 1 == 1", "[landmarks]") {
	REQUIRE(1 == 1);
}

#endif //EOS_LANDMARK_TEST_H
