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
include tools/CMakeFiles/create_label_map.dir/depend.make

# Include the progress variables for this target.
include tools/CMakeFiles/create_label_map.dir/progress.make

# Include the compile flags for this target's objects.
include tools/CMakeFiles/create_label_map.dir/flags.make

tools/CMakeFiles/create_label_map.dir/create_label_map.cpp.o: tools/CMakeFiles/create_label_map.dir/flags.make
tools/CMakeFiles/create_label_map.dir/create_label_map.cpp.o: tools/create_label_map.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/caffe-jacinto/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tools/CMakeFiles/create_label_map.dir/create_label_map.cpp.o"
	cd /workspace/caffe-jacinto/tools && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/create_label_map.dir/create_label_map.cpp.o -c /workspace/caffe-jacinto/tools/create_label_map.cpp

tools/CMakeFiles/create_label_map.dir/create_label_map.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/create_label_map.dir/create_label_map.cpp.i"
	cd /workspace/caffe-jacinto/tools && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/caffe-jacinto/tools/create_label_map.cpp > CMakeFiles/create_label_map.dir/create_label_map.cpp.i

tools/CMakeFiles/create_label_map.dir/create_label_map.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/create_label_map.dir/create_label_map.cpp.s"
	cd /workspace/caffe-jacinto/tools && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/caffe-jacinto/tools/create_label_map.cpp -o CMakeFiles/create_label_map.dir/create_label_map.cpp.s

# Object files for target create_label_map
create_label_map_OBJECTS = \
"CMakeFiles/create_label_map.dir/create_label_map.cpp.o"

# External object files for target create_label_map
create_label_map_EXTERNAL_OBJECTS =

tools/create_label_map: tools/CMakeFiles/create_label_map.dir/create_label_map.cpp.o
tools/create_label_map: tools/CMakeFiles/create_label_map.dir/build.make
tools/create_label_map: lib/libcaffe-nv.so.0.17.0
tools/create_label_map: lib/libproto.a
tools/create_label_map: /root/anaconda3/envs/caffe/lib/libboost_system.so
tools/create_label_map: /root/anaconda3/envs/caffe/lib/libboost_thread.so
tools/create_label_map: /root/anaconda3/envs/caffe/lib/libboost_filesystem.so
tools/create_label_map: /root/anaconda3/envs/caffe/lib/libboost_regex.so
tools/create_label_map: /root/anaconda3/envs/caffe/lib/libboost_chrono.so
tools/create_label_map: /root/anaconda3/envs/caffe/lib/libboost_date_time.so
tools/create_label_map: /root/anaconda3/envs/caffe/lib/libboost_atomic.so
tools/create_label_map: /root/anaconda3/envs/caffe/lib/libglog.so
tools/create_label_map: /root/anaconda3/envs/caffe/lib/libgflags.so
tools/create_label_map: /root/anaconda3/envs/caffe/lib/libprotobuf.so
tools/create_label_map: /root/anaconda3/envs/caffe/lib/libboost_filesystem.so
tools/create_label_map: /root/anaconda3/envs/caffe/lib/libboost_regex.so
tools/create_label_map: /root/anaconda3/envs/caffe/lib/libboost_chrono.so
tools/create_label_map: /root/anaconda3/envs/caffe/lib/libboost_date_time.so
tools/create_label_map: /root/anaconda3/envs/caffe/lib/libboost_atomic.so
tools/create_label_map: /root/anaconda3/envs/caffe/lib/libglog.so
tools/create_label_map: /root/anaconda3/envs/caffe/lib/libgflags.so
tools/create_label_map: /root/anaconda3/envs/caffe/lib/libprotobuf.so
tools/create_label_map: /root/anaconda3/envs/caffe/lib/libhdf5_cpp.so
tools/create_label_map: /root/anaconda3/envs/caffe/lib/libhdf5.so
tools/create_label_map: /usr/lib/x86_64-linux-gnu/librt.so
tools/create_label_map: /usr/lib/x86_64-linux-gnu/libpthread.so
tools/create_label_map: /root/anaconda3/envs/caffe/lib/libz.so
tools/create_label_map: /usr/lib/x86_64-linux-gnu/libdl.so
tools/create_label_map: /usr/lib/x86_64-linux-gnu/libm.so
tools/create_label_map: /root/anaconda3/envs/caffe/lib/libhdf5_hl_cpp.so
tools/create_label_map: /root/anaconda3/envs/caffe/lib/libhdf5_hl.so
tools/create_label_map: /root/anaconda3/envs/caffe/lib/liblmdb.so
tools/create_label_map: /root/anaconda3/envs/caffe/lib/libleveldb.so
tools/create_label_map: /root/anaconda3/envs/caffe/lib/libsnappy.so
tools/create_label_map: /root/anaconda3/envs/caffe/lib/libturbojpeg.so.0
tools/create_label_map: /usr/local/cuda/lib64/libcudart.so
tools/create_label_map: /usr/local/cuda/lib64/libcurand.so
tools/create_label_map: /usr/lib/x86_64-linux-gnu/libcublas.so
tools/create_label_map: /usr/lib/x86_64-linux-gnu/libcudnn.so
tools/create_label_map: /root/anaconda3/envs/caffe/lib/libopencv_highgui.so.3.4.2
tools/create_label_map: /root/anaconda3/envs/caffe/lib/libopencv_videoio.so.3.4.2
tools/create_label_map: /root/anaconda3/envs/caffe/lib/libopencv_imgcodecs.so.3.4.2
tools/create_label_map: /root/anaconda3/envs/caffe/lib/libopencv_imgproc.so.3.4.2
tools/create_label_map: /root/anaconda3/envs/caffe/lib/libopencv_core.so.3.4.2
tools/create_label_map: /root/anaconda3/envs/caffe/lib/libopenblas.so
tools/create_label_map: /root/anaconda3/envs/caffe/lib/libboost_python27.so
tools/create_label_map: /root/anaconda3/envs/caffe/lib/libpython2.7.so
tools/create_label_map: /usr/lib/x86_64-linux-gnu/libnccl.so
tools/create_label_map: /usr/local/cuda/lib64/stubs/libnvidia-ml.so
tools/create_label_map: tools/CMakeFiles/create_label_map.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/caffe-jacinto/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable create_label_map"
	cd /workspace/caffe-jacinto/tools && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/create_label_map.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tools/CMakeFiles/create_label_map.dir/build: tools/create_label_map

.PHONY : tools/CMakeFiles/create_label_map.dir/build

tools/CMakeFiles/create_label_map.dir/clean:
	cd /workspace/caffe-jacinto/tools && $(CMAKE_COMMAND) -P CMakeFiles/create_label_map.dir/cmake_clean.cmake
.PHONY : tools/CMakeFiles/create_label_map.dir/clean

tools/CMakeFiles/create_label_map.dir/depend:
	cd /workspace/caffe-jacinto && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /workspace/caffe-jacinto /workspace/caffe-jacinto/tools /workspace/caffe-jacinto /workspace/caffe-jacinto/tools /workspace/caffe-jacinto/tools/CMakeFiles/create_label_map.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tools/CMakeFiles/create_label_map.dir/depend

