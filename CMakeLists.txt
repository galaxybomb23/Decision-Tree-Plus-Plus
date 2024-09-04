cmake_minimum_required(VERSION 3.10)

project(DecisionTreeProject VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(include)

# Gather source files for the library
file(GLOB_RECURSE LIB_SOURCES "src/*.cpp")
add_library(DecisionTreeLibrary STATIC ${LIB_SOURCES})

# Make the library exportable for use in other projects
target_include_directories(DecisionTreeLibrary PUBLIC include)

# Testing
enable_testing()
find_package(GTest REQUIRED)
file(GLOB_RECURSE TEST_SOURCES "test/*Test.cpp")
add_executable(DecisionTreeTests ${TEST_SOURCES})

# Link the test executable to the library and Google Test
target_link_libraries(DecisionTreeTests DecisionTreeLibrary GTest::gtest_main)
add_test(NAME DecisionTreeTests COMMAND DecisionTreeTests)
