cmake_minimum_required(VERSION 3.10)

project(DecisionTreeProject VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(include)

# Find Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Gather source files for the library
file(GLOB_RECURSE LIB_SOURCES "src/*.cpp")
add_library(DecisionTreeLibrary STATIC ${LIB_SOURCES})

# Make the library exportable for use in other projects
target_include_directories(DecisionTreeLibrary PUBLIC include)

# Link Eigen to the library
target_link_libraries(DecisionTreeLibrary PUBLIC Eigen3::Eigen)

# Testing
enable_testing()
find_package(GTest REQUIRED)
file(GLOB_RECURSE TEST_SOURCES "test/*Test.cpp")
add_executable(DecisionTreeTests ${TEST_SOURCES})

# Link the test executable to the library, Google Test, and Eigen
target_link_libraries(DecisionTreeTests DecisionTreeLibrary GTest::gtest_main Eigen3::Eigen)
add_test(NAME DecisionTreeTests COMMAND DecisionTreeTests)
