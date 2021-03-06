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
include CMakeFiles/pydenseflow.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/pydenseflow.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pydenseflow.dir/flags.make

CMakeFiles/pydenseflow.dir/src/py_denseflow.cpp.o: CMakeFiles/pydenseflow.dir/flags.make
CMakeFiles/pydenseflow.dir/src/py_denseflow.cpp.o: ../src/py_denseflow.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/pydenseflow.dir/src/py_denseflow.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/pydenseflow.dir/src/py_denseflow.cpp.o -c /mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow/src/py_denseflow.cpp

CMakeFiles/pydenseflow.dir/src/py_denseflow.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pydenseflow.dir/src/py_denseflow.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow/src/py_denseflow.cpp > CMakeFiles/pydenseflow.dir/src/py_denseflow.cpp.i

CMakeFiles/pydenseflow.dir/src/py_denseflow.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pydenseflow.dir/src/py_denseflow.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow/src/py_denseflow.cpp -o CMakeFiles/pydenseflow.dir/src/py_denseflow.cpp.s

CMakeFiles/pydenseflow.dir/src/py_denseflow.cpp.o.requires:

.PHONY : CMakeFiles/pydenseflow.dir/src/py_denseflow.cpp.o.requires

CMakeFiles/pydenseflow.dir/src/py_denseflow.cpp.o.provides: CMakeFiles/pydenseflow.dir/src/py_denseflow.cpp.o.requires
	$(MAKE) -f CMakeFiles/pydenseflow.dir/build.make CMakeFiles/pydenseflow.dir/src/py_denseflow.cpp.o.provides.build
.PHONY : CMakeFiles/pydenseflow.dir/src/py_denseflow.cpp.o.provides

CMakeFiles/pydenseflow.dir/src/py_denseflow.cpp.o.provides.build: CMakeFiles/pydenseflow.dir/src/py_denseflow.cpp.o


# Object files for target pydenseflow
pydenseflow_OBJECTS = \
"CMakeFiles/pydenseflow.dir/src/py_denseflow.cpp.o"

# External object files for target pydenseflow
pydenseflow_EXTERNAL_OBJECTS =

libpydenseflow.so: CMakeFiles/pydenseflow.dir/src/py_denseflow.cpp.o
libpydenseflow.so: CMakeFiles/pydenseflow.dir/build.make
libpydenseflow.so: libdenseflow.a
libpydenseflow.so: /usr/lib64/libboost_python-mt.so
libpydenseflow.so: /usr/lib64/libpython2.7.so
libpydenseflow.so: /mnt/151/sch/software/opencv-2.4.13.6/release/lib/libopencv_nonfree.so
libpydenseflow.so: /mnt/151/sch/software/opencv-2.4.13.6/release/lib/libopencv_features2d.so
libpydenseflow.so: /mnt/151/sch/software/opencv-2.4.13.6/release/lib/libopencv_calib3d.so
libpydenseflow.so: /mnt/151/sch/software/opencv-2.4.13.6/release/lib/libopencv_video.so
libpydenseflow.so: /mnt/151/sch/software/opencv-2.4.13.6/release/lib/libopencv_imgproc.so
libpydenseflow.so: /mnt/151/sch/software/opencv-2.4.13.6/release/lib/libopencv_highgui.so
libpydenseflow.so: /mnt/151/sch/software/opencv-2.4.13.6/release/lib/libopencv_core.so
libpydenseflow.so: /mnt/151/sch/software/opencv-2.4.13.6/release/lib/libopencv_gpu.so
libpydenseflow.so: /usr/local/lib64/libzip.so
libpydenseflow.so: CMakeFiles/pydenseflow.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libpydenseflow.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pydenseflow.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pydenseflow.dir/build: libpydenseflow.so

.PHONY : CMakeFiles/pydenseflow.dir/build

CMakeFiles/pydenseflow.dir/requires: CMakeFiles/pydenseflow.dir/src/py_denseflow.cpp.o.requires

.PHONY : CMakeFiles/pydenseflow.dir/requires

CMakeFiles/pydenseflow.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pydenseflow.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pydenseflow.dir/clean

CMakeFiles/pydenseflow.dir/depend:
	cd /mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow /mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow /mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow/build /mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow/build /mnt/151/sch/jhy/3dNet/tools/opticalFlow/dense_flow/build/CMakeFiles/pydenseflow.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pydenseflow.dir/depend

