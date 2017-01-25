// Example file, show to use Catch with multiple files through one main_tests file.
#include "catch.hpp"

TEST_CASE("Test 1 == 1", "[fitting]") {
	REQUIRE(1 == 1);
}