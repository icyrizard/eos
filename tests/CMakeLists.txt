project(tests)
cmake_minimum_required(VERSION 2.8.10)

# The examples need a few additional dependencies (e.g. boost filesystem and OpenCV highgui):

#check installed version in order to include the correct OpenCV libraries
#version variable is defined from project root's CMakeLists
if("${OpenCV_VERSION_MAJOR}$" EQUAL 2)
    message(STATUS "OpenCV 2.x detected")
    find_package(OpenCV 2.4.3 REQUIRED core imgproc highgui)
elseif("${OpenCV_VERSION_MAJOR}$" EQUAL 3)
    message(STATUS "OpenCV 3.x detected - including imgcodecs for compatibility")
    find_package(OpenCV 3.0.0 REQUIRED core imgproc imgcodecs)
endif()
message(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV lib dir found at ${OpenCV_LIB_DIR}")
# This allows us to compile in RelWithDebInfo. It'll use the Release-version of OpenCV:
set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)

if(MSVC)
    # The standard find_package for boost on Win finds the dynamic libs, so for dynamic linking to boost we need to #define:
    add_definitions(-DBOOST_ALL_NO_LIB) # Don't use the automatic library linking by boost with VS (#pragma ...). Instead, we specify everything here in cmake.
    add_definitions(-DBOOST_ALL_DYN_LINK) # Link against the dynamic boost lib - needs to match with the version that find_package finds.
endif()
find_package(Boost 1.50.0 COMPONENTS system filesystem program_options REQUIRED)
if(Boost_FOUND)
    message(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
else(Boost_FOUND)
    message(FATAL_ERROR "Boost not found")
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
endif()

# Includes Catch in the project:
enable_testing(true)  # Enables unit-testing.

# Expose required variable (CATCH_INCLUDE_DIR) to parent scope
set(CATCH_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/3rdparty/Catch/include CACHE INTERNAL "Path to include folder for Catch")

add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})

add_executable (fitting-test fitting/fitting_test.cpp)
target_link_libraries(fitting-test Catch ${OPENCV_LIBS} ${BOOST_LIBRARIES})

add_test (NAME fitting-test COMMAND fitting-test)
install(TARGETS fitting-test DESTINATION tests)


# Simple model fitting (orthographic camera & shape to landmarks) example:
#add_executable(fit-model-simple fit-model-simple.cpp)
#add_executable(fitting-tests )
#target_link_libraries(fit-model-simple ${OpenCV_LIBS} ${Boost_LIBRARIES})

# install target:
#install(TARGETS fit-model-simple DESTINATION bin)
#install(DIRECTORY ${CMAKE_SOURCE_DIR}/examples/data DESTINATION bin)

