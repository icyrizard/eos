#define CATCH_CONFIG_MAIN
#include "catch.hpp"

TEST_CASE("stupid/1=1", "Prove that one equals two" ){
	int one = 1;
	REQUIRE( one == 1 );
}

