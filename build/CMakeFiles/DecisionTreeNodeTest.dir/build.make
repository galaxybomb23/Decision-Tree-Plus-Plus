# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/tom/Desktop/classes/senior_design/Decision-Tree-Plus-Plus

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tom/Desktop/classes/senior_design/Decision-Tree-Plus-Plus/build

# Include any dependencies generated for this target.
include CMakeFiles/DecisionTreeNodeTest.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/DecisionTreeNodeTest.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/DecisionTreeNodeTest.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/DecisionTreeNodeTest.dir/flags.make

CMakeFiles/DecisionTreeNodeTest.dir/test/DecisionTreeNodeTest.cpp.o: CMakeFiles/DecisionTreeNodeTest.dir/flags.make
CMakeFiles/DecisionTreeNodeTest.dir/test/DecisionTreeNodeTest.cpp.o: ../test/DecisionTreeNodeTest.cpp
CMakeFiles/DecisionTreeNodeTest.dir/test/DecisionTreeNodeTest.cpp.o: CMakeFiles/DecisionTreeNodeTest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tom/Desktop/classes/senior_design/Decision-Tree-Plus-Plus/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/DecisionTreeNodeTest.dir/test/DecisionTreeNodeTest.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/DecisionTreeNodeTest.dir/test/DecisionTreeNodeTest.cpp.o -MF CMakeFiles/DecisionTreeNodeTest.dir/test/DecisionTreeNodeTest.cpp.o.d -o CMakeFiles/DecisionTreeNodeTest.dir/test/DecisionTreeNodeTest.cpp.o -c /home/tom/Desktop/classes/senior_design/Decision-Tree-Plus-Plus/test/DecisionTreeNodeTest.cpp

CMakeFiles/DecisionTreeNodeTest.dir/test/DecisionTreeNodeTest.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DecisionTreeNodeTest.dir/test/DecisionTreeNodeTest.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tom/Desktop/classes/senior_design/Decision-Tree-Plus-Plus/test/DecisionTreeNodeTest.cpp > CMakeFiles/DecisionTreeNodeTest.dir/test/DecisionTreeNodeTest.cpp.i

CMakeFiles/DecisionTreeNodeTest.dir/test/DecisionTreeNodeTest.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DecisionTreeNodeTest.dir/test/DecisionTreeNodeTest.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tom/Desktop/classes/senior_design/Decision-Tree-Plus-Plus/test/DecisionTreeNodeTest.cpp -o CMakeFiles/DecisionTreeNodeTest.dir/test/DecisionTreeNodeTest.cpp.s

CMakeFiles/DecisionTreeNodeTest.dir/src/DecisionTreeNode.cpp.o: CMakeFiles/DecisionTreeNodeTest.dir/flags.make
CMakeFiles/DecisionTreeNodeTest.dir/src/DecisionTreeNode.cpp.o: ../src/DecisionTreeNode.cpp
CMakeFiles/DecisionTreeNodeTest.dir/src/DecisionTreeNode.cpp.o: CMakeFiles/DecisionTreeNodeTest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tom/Desktop/classes/senior_design/Decision-Tree-Plus-Plus/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/DecisionTreeNodeTest.dir/src/DecisionTreeNode.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/DecisionTreeNodeTest.dir/src/DecisionTreeNode.cpp.o -MF CMakeFiles/DecisionTreeNodeTest.dir/src/DecisionTreeNode.cpp.o.d -o CMakeFiles/DecisionTreeNodeTest.dir/src/DecisionTreeNode.cpp.o -c /home/tom/Desktop/classes/senior_design/Decision-Tree-Plus-Plus/src/DecisionTreeNode.cpp

CMakeFiles/DecisionTreeNodeTest.dir/src/DecisionTreeNode.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DecisionTreeNodeTest.dir/src/DecisionTreeNode.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tom/Desktop/classes/senior_design/Decision-Tree-Plus-Plus/src/DecisionTreeNode.cpp > CMakeFiles/DecisionTreeNodeTest.dir/src/DecisionTreeNode.cpp.i

CMakeFiles/DecisionTreeNodeTest.dir/src/DecisionTreeNode.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DecisionTreeNodeTest.dir/src/DecisionTreeNode.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tom/Desktop/classes/senior_design/Decision-Tree-Plus-Plus/src/DecisionTreeNode.cpp -o CMakeFiles/DecisionTreeNodeTest.dir/src/DecisionTreeNode.cpp.s

# Object files for target DecisionTreeNodeTest
DecisionTreeNodeTest_OBJECTS = \
"CMakeFiles/DecisionTreeNodeTest.dir/test/DecisionTreeNodeTest.cpp.o" \
"CMakeFiles/DecisionTreeNodeTest.dir/src/DecisionTreeNode.cpp.o"

# External object files for target DecisionTreeNodeTest
DecisionTreeNodeTest_EXTERNAL_OBJECTS =

DecisionTreeNodeTest: CMakeFiles/DecisionTreeNodeTest.dir/test/DecisionTreeNodeTest.cpp.o
DecisionTreeNodeTest: CMakeFiles/DecisionTreeNodeTest.dir/src/DecisionTreeNode.cpp.o
DecisionTreeNodeTest: CMakeFiles/DecisionTreeNodeTest.dir/build.make
DecisionTreeNodeTest: /home/tom/miniconda3/lib/libgtest_main.so.1.11.0
DecisionTreeNodeTest: /home/tom/miniconda3/lib/libgtest.so.1.11.0
DecisionTreeNodeTest: CMakeFiles/DecisionTreeNodeTest.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/tom/Desktop/classes/senior_design/Decision-Tree-Plus-Plus/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable DecisionTreeNodeTest"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/DecisionTreeNodeTest.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/DecisionTreeNodeTest.dir/build: DecisionTreeNodeTest
.PHONY : CMakeFiles/DecisionTreeNodeTest.dir/build

CMakeFiles/DecisionTreeNodeTest.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/DecisionTreeNodeTest.dir/cmake_clean.cmake
.PHONY : CMakeFiles/DecisionTreeNodeTest.dir/clean

CMakeFiles/DecisionTreeNodeTest.dir/depend:
	cd /home/tom/Desktop/classes/senior_design/Decision-Tree-Plus-Plus/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tom/Desktop/classes/senior_design/Decision-Tree-Plus-Plus /home/tom/Desktop/classes/senior_design/Decision-Tree-Plus-Plus /home/tom/Desktop/classes/senior_design/Decision-Tree-Plus-Plus/build /home/tom/Desktop/classes/senior_design/Decision-Tree-Plus-Plus/build /home/tom/Desktop/classes/senior_design/Decision-Tree-Plus-Plus/build/CMakeFiles/DecisionTreeNodeTest.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/DecisionTreeNodeTest.dir/depend

