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
include tools/CMakeFiles/device_query.dir/depend.make

# Include the progress variables for this target.
include tools/CMakeFiles/device_query.dir/progress.make

# Include the compile flags for this target's objects.
include tools/CMakeFiles/device_query.dir/flags.make

tools/CMakeFiles/device_query.dir/device_query.cpp.o: tools/CMakeFiles/device_query.dir/flags.make
tools/CMakeFiles/device_query.dir/device_query.cpp.o: tools/device_query.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/caffe-jacinto/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tools/CMakeFiles/device_query.dir/device_query.cpp.o"
	cd /workspace/caffe-jacinto/tools && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/device_query.dir/device_query.cpp.o -c /workspace/caffe-jacinto/tools/device_query.cpp

tools/CMakeFiles/device_query.dir/device_query.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/device_query.dir/device_query.cpp.i"
	cd /workspace/caffe-jacinto/tools && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/caffe-jacinto/tools/device_query.cpp > CMakeFiles/device_query.dir/device_query.cpp.i

tools/CMakeFiles/device_query.dir/device_query.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/device_query.dir/device_query.cpp.s"
	cd /workspace/caffe-jacinto/tools && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/caffe-jacinto/tools/device_query.cpp -o CMakeFiles/device_query.dir/device_query.cpp.s

# Object files for target device_query
device_query_OBJECTS = \
"CMakeFiles/device_query.dir/device_query.cpp.o"

# External object files for target device_query
device_query_EXTERNAL_OBJECTS =

tools/device_query: tools/CMakeFiles/device_query.dir/device_query.cpp.o
tools/device_query: tools/CMakeFiles/device_query.dir/build.make
tools/device_query: lib/libcaffe-nv.so.0.17.0
tools/device_query: lib/libproto.a
tools/device_query: /root/anaconda3/envs/caffe/lib/libboost_system.so
tools/device_query: /root/anaconda3/envs/caffe/lib/libboost_thread.so
tools/device_query: /root/anaconda3/envs/caffe/lib/libboost_filesystem.so
tools/device_query: /root/anaconda3/envs/caffe/lib/libboost_regex.so
tools/device_query: /root/anaconda3/envs/caffe/lib/libboost_chrono.so
tools/device_query: /root/anaconda3/envs/caffe/lib/libboost_date_time.so
tools/device_query: /root/anaconda3/envs/caffe/lib/libboost_atomic.so
tools/device_query: /root/anaconda3/envs/caffe/lib/libglog.so
tools/device_query: /root/anaconda3/envs/caffe/lib/libgflags.so
tools/device_query: /root/anaconda3/envs/caffe/lib/libprotobuf.so
tools/device_query: /root/anaconda3/envs/caffe/lib/libboost_filesystem.so
tools/device_query: /root/anaconda3/envs/caffe/lib/libboost_regex.so
tools/device_query: /root/anaconda3/envs/caffe/lib/libboost_chrono.so
tools/device_query: /root/anaconda3/envs/caffe/lib/libboost_date_time.so
tools/device_query: /root/anaconda3/envs/caffe/lib/libboost_atomic.so
tools/device_query: /root/anaconda3/envs/caffe/lib/libglog.so
tools/device_query: /root/anaconda3/envs/caffe/lib/libgflags.so
tools/device_query: /root/anaconda3/envs/caffe/lib/libprotobuf.so
tools/device_query: /root/anaconda3/envs/caffe/lib/libhdf5_cpp.so
tools/device_query: /root/anaconda3/envs/caffe/lib/libhdf5.so
tools/device_query: /usr/lib/x86_64-linux-gnu/librt.so
tools/device_query: /usr/lib/x86_64-linux-gnu/libpthread.so
tools/device_query: /root/anaconda3/envs/caffe/lib/libz.so
tools/device_query: /usr/lib/x86_64-linux-gnu/libdl.so
tools/device_query: /usr/lib/x86_64-linux-gnu/libm.so
tools/device_query: /root/anaconda3/envs/caffe/lib/libhdf5_hl_cpp.so
tools/device_query: /root/anaconda3/envs/caffe/lib/libhdf5_hl.so
tools/device_query: /root/anaconda3/envs/caffe/lib/liblmdb.so
tools/device_query: /root/anaconda3/envs/caffe/lib/libleveldb.so
tools/device_query: /root/anaconda3/envs/caffe/lib/libsnappy.so
tools/device_query: /root/anaconda3/envs/caffe/lib/libturbojpeg.so.0
tools/device_query: /usr/local/cuda/lib64/libcudart.so
tools/device_query: /usr/local/cuda/lib64/libcurand.so
tools/device_query: /usr/lib/x86_64-linux-gnu/libcublas.so
tools/device_query: /usr/lib/x86_64-linux-gnu/libcudnn.so
tools/device_query: /root/anaconda3/envs/caffe/lib/libopencv_highgui.so.3.4.2
tools/device_query: /root/anaconda3/envs/caffe/lib/libopencv_videoio.so.3.4.2
tools/device_query: /root/anaconda3/envs/caffe/lib/libopencv_imgcodecs.so.3.4.2
tools/device_query: /root/anaconda3/envs/caffe/lib/libopencv_imgproc.so.3.4.2
tools/device_query: /root/anaconda3/envs/caffe/lib/libopencv_core.so.3.4.2
tools/device_query: /root/anaconda3/envs/caffe/lib/libopenblas.so
tools/device_query: /root/anaconda3/envs/caffe/lib/libboost_python27.so
tools/device_query: /root/anaconda3/envs/caffe/lib/libpython2.7.so
tools/device_query: /usr/lib/x86_64-linux-gnu/libnccl.so
tools/device_query: /usr/local/cuda/lib64/stubs/libnvidia-ml.so
tools/device_query: tools/CMakeFiles/device_query.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/caffe-jacinto/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable device_query"
	cd /workspace/caffe-jacinto/tools && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/device_query.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tools/CMakeFiles/device_query.dir/build: tools/device_query

.PHONY : tools/CMakeFiles/device_query.dir/build

tools/CMakeFiles/device_query.dir/clean:
	cd /workspace/caffe-jacinto/tools && $(CMAKE_COMMAND) -P CMakeFiles/device_query.dir/cmake_clean.cmake
.PHONY : tools/CMakeFiles/device_query.dir/clean

tools/CMakeFiles/device_query.dir/depend:
	cd /workspace/caffe-jacinto && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /workspace/caffe-jacinto /workspace/caffe-jacinto/tools /workspace/caffe-jacinto /workspace/caffe-jacinto/tools /workspace/caffe-jacinto/tools/CMakeFiles/device_query.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tools/CMakeFiles/device_query.dir/depend

