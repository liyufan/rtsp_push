# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

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
CMAKE_COMMAND = /home/ganyi/Downloads/clion-2018.3.4/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/ganyi/Downloads/clion-2018.3.4/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ganyi/vscodeproject/rtsp/rtsp_push

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ganyi/vscodeproject/rtsp/rtsp_push/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/onnx_test.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/onnx_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/onnx_test.dir/flags.make

CMakeFiles/onnx_test.dir/test_onnx.cpp.o: CMakeFiles/onnx_test.dir/flags.make
CMakeFiles/onnx_test.dir/test_onnx.cpp.o: ../test_onnx.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ganyi/vscodeproject/rtsp/rtsp_push/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/onnx_test.dir/test_onnx.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/onnx_test.dir/test_onnx.cpp.o -c /home/ganyi/vscodeproject/rtsp/rtsp_push/test_onnx.cpp

CMakeFiles/onnx_test.dir/test_onnx.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/onnx_test.dir/test_onnx.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ganyi/vscodeproject/rtsp/rtsp_push/test_onnx.cpp > CMakeFiles/onnx_test.dir/test_onnx.cpp.i

CMakeFiles/onnx_test.dir/test_onnx.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/onnx_test.dir/test_onnx.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ganyi/vscodeproject/rtsp/rtsp_push/test_onnx.cpp -o CMakeFiles/onnx_test.dir/test_onnx.cpp.s

# Object files for target onnx_test
onnx_test_OBJECTS = \
"CMakeFiles/onnx_test.dir/test_onnx.cpp.o"

# External object files for target onnx_test
onnx_test_EXTERNAL_OBJECTS =

onnx_test: CMakeFiles/onnx_test.dir/test_onnx.cpp.o
onnx_test: CMakeFiles/onnx_test.dir/build.make
onnx_test: /home/ganyi/opencv-3.4.10/build/lib/libopencv_cudabgsegm.so.3.4.10
onnx_test: /home/ganyi/opencv-3.4.10/build/lib/libopencv_cudaobjdetect.so.3.4.10
onnx_test: /home/ganyi/opencv-3.4.10/build/lib/libopencv_cudastereo.so.3.4.10
onnx_test: /home/ganyi/opencv-3.4.10/build/lib/libopencv_dnn.so.3.4.10
onnx_test: /home/ganyi/opencv-3.4.10/build/lib/libopencv_highgui.so.3.4.10
onnx_test: /home/ganyi/opencv-3.4.10/build/lib/libopencv_ml.so.3.4.10
onnx_test: /home/ganyi/opencv-3.4.10/build/lib/libopencv_shape.so.3.4.10
onnx_test: /home/ganyi/opencv-3.4.10/build/lib/libopencv_stitching.so.3.4.10
onnx_test: /home/ganyi/opencv-3.4.10/build/lib/libopencv_superres.so.3.4.10
onnx_test: /home/ganyi/opencv-3.4.10/build/lib/libopencv_videostab.so.3.4.10
onnx_test: /home/ganyi/opencv-3.4.10/build/lib/libopencv_viz.so.3.4.10
onnx_test: libnmslib.so
onnx_test: /home/ganyi/opencv-3.4.10/build/lib/libopencv_cudafeatures2d.so.3.4.10
onnx_test: /home/ganyi/opencv-3.4.10/build/lib/libopencv_cudacodec.so.3.4.10
onnx_test: /home/ganyi/opencv-3.4.10/build/lib/libopencv_cudaoptflow.so.3.4.10
onnx_test: /home/ganyi/opencv-3.4.10/build/lib/libopencv_cudalegacy.so.3.4.10
onnx_test: /home/ganyi/opencv-3.4.10/build/lib/libopencv_cudawarping.so.3.4.10
onnx_test: /home/ganyi/opencv-3.4.10/build/lib/libopencv_objdetect.so.3.4.10
onnx_test: /home/ganyi/opencv-3.4.10/build/lib/libopencv_calib3d.so.3.4.10
onnx_test: /home/ganyi/opencv-3.4.10/build/lib/libopencv_features2d.so.3.4.10
onnx_test: /home/ganyi/opencv-3.4.10/build/lib/libopencv_flann.so.3.4.10
onnx_test: /home/ganyi/opencv-3.4.10/build/lib/libopencv_photo.so.3.4.10
onnx_test: /home/ganyi/opencv-3.4.10/build/lib/libopencv_cudaimgproc.so.3.4.10
onnx_test: /home/ganyi/opencv-3.4.10/build/lib/libopencv_cudafilters.so.3.4.10
onnx_test: /home/ganyi/opencv-3.4.10/build/lib/libopencv_cudaarithm.so.3.4.10
onnx_test: /home/ganyi/opencv-3.4.10/build/lib/libopencv_video.so.3.4.10
onnx_test: /home/ganyi/opencv-3.4.10/build/lib/libopencv_videoio.so.3.4.10
onnx_test: /home/ganyi/opencv-3.4.10/build/lib/libopencv_imgcodecs.so.3.4.10
onnx_test: /home/ganyi/opencv-3.4.10/build/lib/libopencv_imgproc.so.3.4.10
onnx_test: /home/ganyi/opencv-3.4.10/build/lib/libopencv_core.so.3.4.10
onnx_test: /home/ganyi/opencv-3.4.10/build/lib/libopencv_cudev.so.3.4.10
onnx_test: /usr/local/cuda/lib64/libcudart.so
onnx_test: CMakeFiles/onnx_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ganyi/vscodeproject/rtsp/rtsp_push/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable onnx_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/onnx_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/onnx_test.dir/build: onnx_test

.PHONY : CMakeFiles/onnx_test.dir/build

CMakeFiles/onnx_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/onnx_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/onnx_test.dir/clean

CMakeFiles/onnx_test.dir/depend:
	cd /home/ganyi/vscodeproject/rtsp/rtsp_push/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ganyi/vscodeproject/rtsp/rtsp_push /home/ganyi/vscodeproject/rtsp/rtsp_push /home/ganyi/vscodeproject/rtsp/rtsp_push/cmake-build-debug /home/ganyi/vscodeproject/rtsp/rtsp_push/cmake-build-debug /home/ganyi/vscodeproject/rtsp/rtsp_push/cmake-build-debug/CMakeFiles/onnx_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/onnx_test.dir/depend

