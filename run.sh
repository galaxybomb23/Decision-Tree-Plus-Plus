#!/bin/bash
# Script to build and test the project, not to be considered as a part of the project
# Written with help of ChatGPT for argument parsing

# "defensive Programming"
cd "$(dirname "$0")"

# Function to perform the build
build() {
    echo "Building..."
    
    if [ ! -d "build" ]; then
        mkdir build
    fi
    
    cd build
    cmake ..
    make
    cd ..
}

#remove build repo
clean(){
    echo "Cleaning build..."
    rm -r build 
}
# Function to perform the tests
test() {
    build
    echo "Running tests..."
    cd build
    ctest --rerun-failed --output-on-failure
    cd ..
}

# Parse command-line arguments
while getopts "ct" opt; do
    case $opt in
        c)
            clean_flag=true
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

if [ "$clean_flag" = true ]; then
    clean
fi
# Execute test if requested
if [ "$test_flag" = true ]; then
    test
    exit 0
fi

build