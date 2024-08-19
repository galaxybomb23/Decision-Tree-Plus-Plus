#!/bin/bash
# Script to build and test the project, not to be considered as a part of the project
# Written with help of ChatGPT for argument parsing

# Function to perform the build
build() {
    echo "Building..."
    cd build
    rm -rf *
    cmake ..
    make
}

# Function to perform the tests
test() {
    echo "Running tests..."
    cd build
    ctest
}

# Parse command-line arguments
while getopts "bt" opt; do
    case $opt in
        b)
            build_flag=true
            ;;
        t)
            test_flag=true
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
    esac
done

# Execute build if requested
if [ "$build_flag" = true ]; then
    build
fi

# Execute test if requested
if [ "$test_flag" = true ]; then
    test
fi

# If no options were provided, show usage
if [ -z "$build_flag" ] && [ -z "$test_flag" ]; then
    echo "Usage: $0 [-b] [-t]"
    echo "  -b    Build"
    echo "  -t    Test"
    exit 1
fi
