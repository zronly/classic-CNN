# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.3

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow/build

# Include any dependencies generated for this target.
include CMakeFiles/denseflow.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/denseflow.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/denseflow.dir/flags.make

CMakeFiles/denseflow.dir/src/common.cpp.o: CMakeFiles/denseflow.dir/flags.make
CMakeFiles/denseflow.dir/src/common.cpp.o: ../src/common.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/denseflow.dir/src/common.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/denseflow.dir/src/common.cpp.o -c /mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow/src/common.cpp

CMakeFiles/denseflow.dir/src/common.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/denseflow.dir/src/common.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow/src/common.cpp > CMakeFiles/denseflow.dir/src/common.cpp.i

CMakeFiles/denseflow.dir/src/common.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/denseflow.dir/src/common.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow/src/common.cpp -o CMakeFiles/denseflow.dir/src/common.cpp.s

CMakeFiles/denseflow.dir/src/common.cpp.o.requires:

.PHONY : CMakeFiles/denseflow.dir/src/common.cpp.o.requires

CMakeFiles/denseflow.dir/src/common.cpp.o.provides: CMakeFiles/denseflow.dir/src/common.cpp.o.requires
	$(MAKE) -f CMakeFiles/denseflow.dir/build.make CMakeFiles/denseflow.dir/src/common.cpp.o.provides.build
.PHONY : CMakeFiles/denseflow.dir/src/common.cpp.o.provides

CMakeFiles/denseflow.dir/src/common.cpp.o.provides.build: CMakeFiles/denseflow.dir/src/common.cpp.o


CMakeFiles/denseflow.dir/src/dense_flow.cpp.o: CMakeFiles/denseflow.dir/flags.make
CMakeFiles/denseflow.dir/src/dense_flow.cpp.o: ../src/dense_flow.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/denseflow.dir/src/dense_flow.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/denseflow.dir/src/dense_flow.cpp.o -c /mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow/src/dense_flow.cpp

CMakeFiles/denseflow.dir/src/dense_flow.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/denseflow.dir/src/dense_flow.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow/src/dense_flow.cpp > CMakeFiles/denseflow.dir/src/dense_flow.cpp.i

CMakeFiles/denseflow.dir/src/dense_flow.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/denseflow.dir/src/dense_flow.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow/src/dense_flow.cpp -o CMakeFiles/denseflow.dir/src/dense_flow.cpp.s

CMakeFiles/denseflow.dir/src/dense_flow.cpp.o.requires:

.PHONY : CMakeFiles/denseflow.dir/src/dense_flow.cpp.o.requires

CMakeFiles/denseflow.dir/src/dense_flow.cpp.o.provides: CMakeFiles/denseflow.dir/src/dense_flow.cpp.o.requires
	$(MAKE) -f CMakeFiles/denseflow.dir/build.make CMakeFiles/denseflow.dir/src/dense_flow.cpp.o.provides.build
.PHONY : CMakeFiles/denseflow.dir/src/dense_flow.cpp.o.provides

CMakeFiles/denseflow.dir/src/dense_flow.cpp.o.provides.build: CMakeFiles/denseflow.dir/src/dense_flow.cpp.o


CMakeFiles/denseflow.dir/src/dense_flow_gpu.cpp.o: CMakeFiles/denseflow.dir/flags.make
CMakeFiles/denseflow.dir/src/dense_flow_gpu.cpp.o: ../src/dense_flow_gpu.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/denseflow.dir/src/dense_flow_gpu.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/denseflow.dir/src/dense_flow_gpu.cpp.o -c /mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow/src/dense_flow_gpu.cpp

CMakeFiles/denseflow.dir/src/dense_flow_gpu.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/denseflow.dir/src/dense_flow_gpu.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow/src/dense_flow_gpu.cpp > CMakeFiles/denseflow.dir/src/dense_flow_gpu.cpp.i

CMakeFiles/denseflow.dir/src/dense_flow_gpu.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/denseflow.dir/src/dense_flow_gpu.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow/src/dense_flow_gpu.cpp -o CMakeFiles/denseflow.dir/src/dense_flow_gpu.cpp.s

CMakeFiles/denseflow.dir/src/dense_flow_gpu.cpp.o.requires:

.PHONY : CMakeFiles/denseflow.dir/src/dense_flow_gpu.cpp.o.requires

CMakeFiles/denseflow.dir/src/dense_flow_gpu.cpp.o.provides: CMakeFiles/denseflow.dir/src/dense_flow_gpu.cpp.o.requires
	$(MAKE) -f CMakeFiles/denseflow.dir/build.make CMakeFiles/denseflow.dir/src/dense_flow_gpu.cpp.o.provides.build
.PHONY : CMakeFiles/denseflow.dir/src/dense_flow_gpu.cpp.o.provides

CMakeFiles/denseflow.dir/src/dense_flow_gpu.cpp.o.provides.build: CMakeFiles/denseflow.dir/src/dense_flow_gpu.cpp.o


CMakeFiles/denseflow.dir/src/dense_warp_flow_gpu.cpp.o: CMakeFiles/denseflow.dir/flags.make
CMakeFiles/denseflow.dir/src/dense_warp_flow_gpu.cpp.o: ../src/dense_warp_flow_gpu.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/denseflow.dir/src/dense_warp_flow_gpu.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/denseflow.dir/src/dense_warp_flow_gpu.cpp.o -c /mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow/src/dense_warp_flow_gpu.cpp

CMakeFiles/denseflow.dir/src/dense_warp_flow_gpu.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/denseflow.dir/src/dense_warp_flow_gpu.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow/src/dense_warp_flow_gpu.cpp > CMakeFiles/denseflow.dir/src/dense_warp_flow_gpu.cpp.i

CMakeFiles/denseflow.dir/src/dense_warp_flow_gpu.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/denseflow.dir/src/dense_warp_flow_gpu.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow/src/dense_warp_flow_gpu.cpp -o CMakeFiles/denseflow.dir/src/dense_warp_flow_gpu.cpp.s

CMakeFiles/denseflow.dir/src/dense_warp_flow_gpu.cpp.o.requires:

.PHONY : CMakeFiles/denseflow.dir/src/dense_warp_flow_gpu.cpp.o.requires

CMakeFiles/denseflow.dir/src/dense_warp_flow_gpu.cpp.o.provides: CMakeFiles/denseflow.dir/src/dense_warp_flow_gpu.cpp.o.requires
	$(MAKE) -f CMakeFiles/denseflow.dir/build.make CMakeFiles/denseflow.dir/src/dense_warp_flow_gpu.cpp.o.provides.build
.PHONY : CMakeFiles/denseflow.dir/src/dense_warp_flow_gpu.cpp.o.provides

CMakeFiles/denseflow.dir/src/dense_warp_flow_gpu.cpp.o.provides.build: CMakeFiles/denseflow.dir/src/dense_warp_flow_gpu.cpp.o


CMakeFiles/denseflow.dir/src/zip_utils.cpp.o: CMakeFiles/denseflow.dir/flags.make
CMakeFiles/denseflow.dir/src/zip_utils.cpp.o: ../src/zip_utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/denseflow.dir/src/zip_utils.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/denseflow.dir/src/zip_utils.cpp.o -c /mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow/src/zip_utils.cpp

CMakeFiles/denseflow.dir/src/zip_utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/denseflow.dir/src/zip_utils.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow/src/zip_utils.cpp > CMakeFiles/denseflow.dir/src/zip_utils.cpp.i

CMakeFiles/denseflow.dir/src/zip_utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/denseflow.dir/src/zip_utils.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow/src/zip_utils.cpp -o CMakeFiles/denseflow.dir/src/zip_utils.cpp.s

CMakeFiles/denseflow.dir/src/zip_utils.cpp.o.requires:

.PHONY : CMakeFiles/denseflow.dir/src/zip_utils.cpp.o.requires

CMakeFiles/denseflow.dir/src/zip_utils.cpp.o.provides: CMakeFiles/denseflow.dir/src/zip_utils.cpp.o.requires
	$(MAKE) -f CMakeFiles/denseflow.dir/build.make CMakeFiles/denseflow.dir/src/zip_utils.cpp.o.provides.build
.PHONY : CMakeFiles/denseflow.dir/src/zip_utils.cpp.o.provides

CMakeFiles/denseflow.dir/src/zip_utils.cpp.o.provides.build: CMakeFiles/denseflow.dir/src/zip_utils.cpp.o


# Object files for target denseflow
denseflow_OBJECTS = \
"CMakeFiles/denseflow.dir/src/common.cpp.o" \
"CMakeFiles/denseflow.dir/src/dense_flow.cpp.o" \
"CMakeFiles/denseflow.dir/src/dense_flow_gpu.cpp.o" \
"CMakeFiles/denseflow.dir/src/dense_warp_flow_gpu.cpp.o" \
"CMakeFiles/denseflow.dir/src/zip_utils.cpp.o"

# External object files for target denseflow
denseflow_EXTERNAL_OBJECTS =

libdenseflow.a: CMakeFiles/denseflow.dir/src/common.cpp.o
libdenseflow.a: CMakeFiles/denseflow.dir/src/dense_flow.cpp.o
libdenseflow.a: CMakeFiles/denseflow.dir/src/dense_flow_gpu.cpp.o
libdenseflow.a: CMakeFiles/denseflow.dir/src/dense_warp_flow_gpu.cpp.o
libdenseflow.a: CMakeFiles/denseflow.dir/src/zip_utils.cpp.o
libdenseflow.a: CMakeFiles/denseflow.dir/build.make
libdenseflow.a: CMakeFiles/denseflow.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX static library libdenseflow.a"
	$(CMAKE_COMMAND) -P CMakeFiles/denseflow.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/denseflow.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/denseflow.dir/build: libdenseflow.a

.PHONY : CMakeFiles/denseflow.dir/build

CMakeFiles/denseflow.dir/requires: CMakeFiles/denseflow.dir/src/common.cpp.o.requires
CMakeFiles/denseflow.dir/requires: CMakeFiles/denseflow.dir/src/dense_flow.cpp.o.requires
CMakeFiles/denseflow.dir/requires: CMakeFiles/denseflow.dir/src/dense_flow_gpu.cpp.o.requires
CMakeFiles/denseflow.dir/requires: CMakeFiles/denseflow.dir/src/dense_warp_flow_gpu.cpp.o.requires
CMakeFiles/denseflow.dir/requires: CMakeFiles/denseflow.dir/src/zip_utils.cpp.o.requires

.PHONY : CMakeFiles/denseflow.dir/requires

CMakeFiles/denseflow.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/denseflow.dir/cmake_clean.cmake
.PHONY : CMakeFiles/denseflow.dir/clean

CMakeFiles/denseflow.dir/depend:
	cd /mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow /mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow /mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow/build /mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow/build /mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow/build/CMakeFiles/denseflow.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/denseflow.dir/depend

