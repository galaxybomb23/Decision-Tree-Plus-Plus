cmake_minimum_required(VERSION 3.10)

project(DecisionTreeProject VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_BUILD_TYPE Release)

include_directories(include)

# Find Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Gather source files for the library
file(GLOB_RECURSE LIB_SOURCES "src/*.cpp")

# Exclude test files and from the library sources
list(FILTER LIB_SOURCES EXCLUDE REGEX ".*test/.*")

add_library(DecisionTreeLibrary STATIC ${LIB_SOURCES})

# Make the library exportable for use in other projects
target_include_directories(DecisionTreeLibrary
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Link Eigen to the library
target_link_libraries(DecisionTreeLibrary PUBLIC Eigen3::Eigen)

# Set up installation paths
include(GNUInstallDirs)
install(TARGETS DecisionTreeLibrary
        EXPORT DecisionTreeLibraryConfig
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Export library configuration for use in other projects
install(EXPORT DecisionTreeLibraryConfig
        NAMESPACE DecisionTree::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DecisionTreeLibrary
)

enable_testing()
find_package(GTest REQUIRED)

# Get all test files individually
file(GLOB TEST_SOURCES "test/*Test.cpp")

# Create a separate test executable for each test file
foreach(TEST_SOURCE ${TEST_SOURCES})
        # Get the filename without extension
        get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)

        # Create executable for this test
        add_executable(${TEST_NAME} ${TEST_SOURCE})

        # Link libraries
        target_link_libraries(${TEST_NAME}
                DecisionTreeLibrary
                GTest::gtest_main
                Eigen3::Eigen
        )

        # Add the test
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()

# **Add the Sandbox executable**
add_executable(Sandbox test/Sandbox.cpp)

# Link the Sandbox executable to the library and Eigen
target_link_libraries(Sandbox DecisionTreeLibrary Eigen3::Eigen)
