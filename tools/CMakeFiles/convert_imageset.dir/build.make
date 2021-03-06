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
include tools/CMakeFiles/convert_imageset.dir/depend.make

# Include the progress variables for this target.
include tools/CMakeFiles/convert_imageset.dir/progress.make

# Include the compile flags for this target's objects.
include tools/CMakeFiles/convert_imageset.dir/flags.make

tools/CMakeFiles/convert_imageset.dir/convert_imageset.cpp.o: tools/CMakeFiles/convert_imageset.dir/flags.make
tools/CMakeFiles/convert_imageset.dir/convert_imageset.cpp.o: tools/convert_imageset.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/caffe-jacinto/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tools/CMakeFiles/convert_imageset.dir/convert_imageset.cpp.o"
	cd /workspace/caffe-jacinto/tools && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/convert_imageset.dir/convert_imageset.cpp.o -c /workspace/caffe-jacinto/tools/convert_imageset.cpp

tools/CMakeFiles/convert_imageset.dir/convert_imageset.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/convert_imageset.dir/convert_imageset.cpp.i"
	cd /workspace/caffe-jacinto/tools && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/caffe-jacinto/tools/convert_imageset.cpp > CMakeFiles/convert_imageset.dir/convert_imageset.cpp.i

tools/CMakeFiles/convert_imageset.dir/convert_imageset.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/convert_imageset.dir/convert_imageset.cpp.s"
	cd /workspace/caffe-jacinto/tools && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/caffe-jacinto/tools/convert_imageset.cpp -o CMakeFiles/convert_imageset.dir/convert_imageset.cpp.s

# Object files for target convert_imageset
convert_imageset_OBJECTS = \
"CMakeFiles/convert_imageset.dir/convert_imageset.cpp.o"

# External object files for target convert_imageset
convert_imageset_EXTERNAL_OBJECTS =

tools/convert_imageset: tools/CMakeFiles/convert_imageset.dir/convert_imageset.cpp.o
tools/convert_imageset: tools/CMakeFiles/convert_imageset.dir/build.make
tools/convert_imageset: lib/libcaffe-nv.so.0.17.0
tools/convert_imageset: lib/libproto.a
tools/convert_imageset: /root/anaconda3/envs/caffe/lib/libboost_system.so
tools/convert_imageset: /root/anaconda3/envs/caffe/lib/libboost_thread.so
tools/convert_imageset: /root/anaconda3/envs/caffe/lib/libboost_filesystem.so
tools/convert_imageset: /root/anaconda3/envs/caffe/lib/libboost_regex.so
tools/convert_imageset: /root/anaconda3/envs/caffe/lib/libboost_chrono.so
tools/convert_imageset: /root/anaconda3/envs/caffe/lib/libboost_date_time.so
tools/convert_imageset: /root/anaconda3/envs/caffe/lib/libboost_atomic.so
tools/convert_imageset: /root/anaconda3/envs/caffe/lib/libglog.so
tools/convert_imageset: /root/anaconda3/envs/caffe/lib/libgflags.so
tools/convert_imageset: /root/anaconda3/envs/caffe/lib/libprotobuf.so
tools/convert_imageset: /root/anaconda3/envs/caffe/lib/libboost_filesystem.so
tools/convert_imageset: /root/anaconda3/envs/caffe/lib/libboost_regex.so
tools/convert_imageset: /root/anaconda3/envs/caffe/lib/libboost_chrono.so
tools/convert_imageset: /root/anaconda3/envs/caffe/lib/libboost_date_time.so
tools/convert_imageset: /root/anaconda3/envs/caffe/lib/libboost_atomic.so
tools/convert_imageset: /root/anaconda3/envs/caffe/lib/libglog.so
tools/convert_imageset: /root/anaconda3/envs/caffe/lib/libgflags.so
tools/convert_imageset: /root/anaconda3/envs/caffe/lib/libprotobuf.so
tools/convert_imageset: /root/anaconda3/envs/caffe/lib/libhdf5_cpp.so
tools/convert_imageset: /root/anaconda3/envs/caffe/lib/libhdf5.so
tools/convert_imageset: /usr/lib/x86_64-linux-gnu/librt.so
tools/convert_imageset: /usr/lib/x86_64-linux-gnu/libpthread.so
tools/convert_imageset: /root/anaconda3/envs/caffe/lib/libz.so
tools/convert_imageset: /usr/lib/x86_64-linux-gnu/libdl.so
tools/convert_imageset: /usr/lib/x86_64-linux-gnu/libm.so
tools/convert_imageset: /root/anaconda3/envs/caffe/lib/libhdf5_hl_cpp.so
tools/convert_imageset: /root/anaconda3/envs/caffe/lib/libhdf5_hl.so
tools/convert_imageset: /root/anaconda3/envs/caffe/lib/liblmdb.so
tools/convert_imageset: /root/anaconda3/envs/caffe/lib/libleveldb.so
tools/convert_imageset: /root/anaconda3/envs/caffe/lib/libsnappy.so
tools/convert_imageset: /root/anaconda3/envs/caffe/lib/libturbojpeg.so.0
tools/convert_imageset: /usr/local/cuda/lib64/libcudart.so
tools/convert_imageset: /usr/local/cuda/lib64/libcurand.so
tools/convert_imageset: /usr/lib/x86_64-linux-gnu/libcublas.so
tools/convert_imageset: /usr/lib/x86_64-linux-gnu/libcudnn.so
tools/convert_imageset: /root/anaconda3/envs/caffe/lib/libopencv_highgui.so.3.4.2
tools/convert_imageset: /root/anaconda3/envs/caffe/lib/libopencv_videoio.so.3.4.2
tools/convert_imageset: /root/anaconda3/envs/caffe/lib/libopencv_imgcodecs.so.3.4.2
tools/convert_imageset: /root/anaconda3/envs/caffe/lib/libopencv_imgproc.so.3.4.2
tools/convert_imageset: /root/anaconda3/envs/caffe/lib/libopencv_core.so.3.4.2
tools/convert_imageset: /root/anaconda3/envs/caffe/lib/libopenblas.so
tools/convert_imageset: /root/anaconda3/envs/caffe/lib/libboost_python27.so
tools/convert_imageset: /root/anaconda3/envs/caffe/lib/libpython2.7.so
tools/convert_imageset: /usr/lib/x86_64-linux-gnu/libnccl.so
tools/convert_imageset: /usr/local/cuda/lib64/stubs/libnvidia-ml.so
tools/convert_imageset: tools/CMakeFiles/convert_imageset.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/caffe-jacinto/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable convert_imageset"
	cd /workspace/caffe-jacinto/tools && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/convert_imageset.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tools/CMakeFiles/convert_imageset.dir/build: tools/convert_imageset

.PHONY : tools/CMakeFiles/convert_imageset.dir/build

tools/CMakeFiles/convert_imageset.dir/clean:
	cd /workspace/caffe-jacinto/tools && $(CMAKE_COMMAND) -P CMakeFiles/convert_imageset.dir/cmake_clean.cmake
.PHONY : tools/CMakeFiles/convert_imageset.dir/clean

tools/CMakeFiles/convert_imageset.dir/depend:
	cd /workspace/caffe-jacinto && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /workspace/caffe-jacinto /workspace/caffe-jacinto/tools /workspace/caffe-jacinto /workspace/caffe-jacinto/tools /workspace/caffe-jacinto/tools/CMakeFiles/convert_imageset.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tools/CMakeFiles/convert_imageset.dir/depend

