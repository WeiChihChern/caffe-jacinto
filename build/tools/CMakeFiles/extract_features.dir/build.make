# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

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
CMAKE_COMMAND = /root/anaconda3/envs/caffe/bin/cmake

# The command to remove a file.
RM = /root/anaconda3/envs/caffe/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /workspace/caffe-jacinto

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /workspace/caffe-jacinto

# Include any dependencies generated for this target.
include tools/CMakeFiles/extract_features.dir/depend.make

# Include the progress variables for this target.
include tools/CMakeFiles/extract_features.dir/progress.make

# Include the compile flags for this target's objects.
include tools/CMakeFiles/extract_features.dir/flags.make

tools/CMakeFiles/extract_features.dir/extract_features.cpp.o: tools/CMakeFiles/extract_features.dir/flags.make
tools/CMakeFiles/extract_features.dir/extract_features.cpp.o: tools/extract_features.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/caffe-jacinto/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tools/CMakeFiles/extract_features.dir/extract_features.cpp.o"
	cd /workspace/caffe-jacinto/tools && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/extract_features.dir/extract_features.cpp.o -c /workspace/caffe-jacinto/tools/extract_features.cpp

tools/CMakeFiles/extract_features.dir/extract_features.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/extract_features.dir/extract_features.cpp.i"
	cd /workspace/caffe-jacinto/tools && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/caffe-jacinto/tools/extract_features.cpp > CMakeFiles/extract_features.dir/extract_features.cpp.i

tools/CMakeFiles/extract_features.dir/extract_features.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/extract_features.dir/extract_features.cpp.s"
	cd /workspace/caffe-jacinto/tools && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/caffe-jacinto/tools/extract_features.cpp -o CMakeFiles/extract_features.dir/extract_features.cpp.s

# Object files for target extract_features
extract_features_OBJECTS = \
"CMakeFiles/extract_features.dir/extract_features.cpp.o"

# External object files for target extract_features
extract_features_EXTERNAL_OBJECTS =

tools/extract_features: tools/CMakeFiles/extract_features.dir/extract_features.cpp.o
tools/extract_features: tools/CMakeFiles/extract_features.dir/build.make
tools/extract_features: lib/libcaffe-nv.so.0.17.0
tools/extract_features: lib/libproto.a
tools/extract_features: /root/anaconda3/envs/caffe/lib/libboost_system.so
tools/extract_features: /root/anaconda3/envs/caffe/lib/libboost_thread.so
tools/extract_features: /root/anaconda3/envs/caffe/lib/libboost_filesystem.so
tools/extract_features: /root/anaconda3/envs/caffe/lib/libboost_regex.so
tools/extract_features: /root/anaconda3/envs/caffe/lib/libboost_chrono.so
tools/extract_features: /root/anaconda3/envs/caffe/lib/libboost_date_time.so
tools/extract_features: /root/anaconda3/envs/caffe/lib/libboost_atomic.so
tools/extract_features: /root/anaconda3/envs/caffe/lib/libglog.so
tools/extract_features: /root/anaconda3/envs/caffe/lib/libgflags.so
tools/extract_features: /root/anaconda3/envs/caffe/lib/libprotobuf.so
tools/extract_features: /root/anaconda3/envs/caffe/lib/libboost_filesystem.so
tools/extract_features: /root/anaconda3/envs/caffe/lib/libboost_regex.so
tools/extract_features: /root/anaconda3/envs/caffe/lib/libboost_chrono.so
tools/extract_features: /root/anaconda3/envs/caffe/lib/libboost_date_time.so
tools/extract_features: /root/anaconda3/envs/caffe/lib/libboost_atomic.so
tools/extract_features: /root/anaconda3/envs/caffe/lib/libglog.so
tools/extract_features: /root/anaconda3/envs/caffe/lib/libgflags.so
tools/extract_features: /root/anaconda3/envs/caffe/lib/libprotobuf.so
tools/extract_features: /root/anaconda3/envs/caffe/lib/libhdf5_cpp.so
tools/extract_features: /root/anaconda3/envs/caffe/lib/libhdf5.so
tools/extract_features: /usr/lib/x86_64-linux-gnu/librt.so
tools/extract_features: /usr/lib/x86_64-linux-gnu/libpthread.so
tools/extract_features: /root/anaconda3/envs/caffe/lib/libz.so
tools/extract_features: /usr/lib/x86_64-linux-gnu/libdl.so
tools/extract_features: /usr/lib/x86_64-linux-gnu/libm.so
tools/extract_features: /root/anaconda3/envs/caffe/lib/libhdf5_hl_cpp.so
tools/extract_features: /root/anaconda3/envs/caffe/lib/libhdf5_hl.so
tools/extract_features: /root/anaconda3/envs/caffe/lib/liblmdb.so
tools/extract_features: /root/anaconda3/envs/caffe/lib/libleveldb.so
tools/extract_features: /root/anaconda3/envs/caffe/lib/libsnappy.so
tools/extract_features: /root/anaconda3/envs/caffe/lib/libturbojpeg.so.0
tools/extract_features: /usr/local/cuda/lib64/libcudart.so
tools/extract_features: /usr/local/cuda/lib64/libcurand.so
tools/extract_features: /usr/lib/x86_64-linux-gnu/libcublas.so
tools/extract_features: /usr/lib/x86_64-linux-gnu/libcudnn.so
tools/extract_features: /root/anaconda3/envs/caffe/lib/libopencv_highgui.so.3.4.2
tools/extract_features: /root/anaconda3/envs/caffe/lib/libopencv_videoio.so.3.4.2
tools/extract_features: /root/anaconda3/envs/caffe/lib/libopencv_imgcodecs.so.3.4.2
tools/extract_features: /root/anaconda3/envs/caffe/lib/libopencv_imgproc.so.3.4.2
tools/extract_features: /root/anaconda3/envs/caffe/lib/libopencv_core.so.3.4.2
tools/extract_features: /root/anaconda3/envs/caffe/lib/libopenblas.so
tools/extract_features: /root/anaconda3/envs/caffe/lib/libboost_python27.so
tools/extract_features: /root/anaconda3/envs/caffe/lib/libpython2.7.so
tools/extract_features: /usr/lib/x86_64-linux-gnu/libnccl.so
tools/extract_features: /usr/local/cuda/lib64/stubs/libnvidia-ml.so
tools/extract_features: tools/CMakeFiles/extract_features.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/caffe-jacinto/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable extract_features"
	cd /workspace/caffe-jacinto/tools && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/extract_features.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tools/CMakeFiles/extract_features.dir/build: tools/extract_features

.PHONY : tools/CMakeFiles/extract_features.dir/build

tools/CMakeFiles/extract_features.dir/clean:
	cd /workspace/caffe-jacinto/tools && $(CMAKE_COMMAND) -P CMakeFiles/extract_features.dir/cmake_clean.cmake
.PHONY : tools/CMakeFiles/extract_features.dir/clean

tools/CMakeFiles/extract_features.dir/depend:
	cd /workspace/caffe-jacinto && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /workspace/caffe-jacinto /workspace/caffe-jacinto/tools /workspace/caffe-jacinto /workspace/caffe-jacinto/tools /workspace/caffe-jacinto/tools/CMakeFiles/extract_features.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tools/CMakeFiles/extract_features.dir/depend

