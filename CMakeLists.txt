cmake_minimum_required(VERSION 3.10)
project(DecisionTreeProject)
set(CMAKE_CXX_STANDARD 17)

# SOURCES
file(GLOB SOURCES "src/*.cpp")

# HEADERS
include_directories(include)
file(GLOB HEADERS "include/*.hpp")

## TESTS ##
# Link Google Test libraries
find_package(GTest REQUIRED)
enable_testing()

# ADD TEST EXECUTABLES AND LINK LIBRARIES
add_executable(DecisionTreeNodeTest test/DecisionTreeNodeTest.cpp ${SOURCES} ${HEADERS})
target_link_libraries(DecisionTreeNodeTest GTest::gtest_main)

# ADD TESTS
add_test(NAME DecisionTreeNodeTest COMMAND DecisionTreeNodeTest)
