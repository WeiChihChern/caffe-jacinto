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
include tools/CMakeFiles/finetune_net.dir/depend.make

# Include the progress variables for this target.
include tools/CMakeFiles/finetune_net.dir/progress.make

# Include the compile flags for this target's objects.
include tools/CMakeFiles/finetune_net.dir/flags.make

tools/CMakeFiles/finetune_net.dir/finetune_net.cpp.o: tools/CMakeFiles/finetune_net.dir/flags.make
tools/CMakeFiles/finetune_net.dir/finetune_net.cpp.o: tools/finetune_net.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/caffe-jacinto/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tools/CMakeFiles/finetune_net.dir/finetune_net.cpp.o"
	cd /workspace/caffe-jacinto/tools && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/finetune_net.dir/finetune_net.cpp.o -c /workspace/caffe-jacinto/tools/finetune_net.cpp

tools/CMakeFiles/finetune_net.dir/finetune_net.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/finetune_net.dir/finetune_net.cpp.i"
	cd /workspace/caffe-jacinto/tools && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/caffe-jacinto/tools/finetune_net.cpp > CMakeFiles/finetune_net.dir/finetune_net.cpp.i

tools/CMakeFiles/finetune_net.dir/finetune_net.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/finetune_net.dir/finetune_net.cpp.s"
	cd /workspace/caffe-jacinto/tools && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/caffe-jacinto/tools/finetune_net.cpp -o CMakeFiles/finetune_net.dir/finetune_net.cpp.s

# Object files for target finetune_net
finetune_net_OBJECTS = \
"CMakeFiles/finetune_net.dir/finetune_net.cpp.o"

# External object files for target finetune_net
finetune_net_EXTERNAL_OBJECTS =

tools/finetune_net: tools/CMakeFiles/finetune_net.dir/finetune_net.cpp.o
tools/finetune_net: tools/CMakeFiles/finetune_net.dir/build.make
tools/finetune_net: lib/libcaffe-nv.so.0.17.0
tools/finetune_net: lib/libproto.a
tools/finetune_net: /root/anaconda3/envs/caffe/lib/libboost_system.so
tools/finetune_net: /root/anaconda3/envs/caffe/lib/libboost_thread.so
tools/finetune_net: /root/anaconda3/envs/caffe/lib/libboost_filesystem.so
tools/finetune_net: /root/anaconda3/envs/caffe/lib/libboost_regex.so
tools/finetune_net: /root/anaconda3/envs/caffe/lib/libboost_chrono.so
tools/finetune_net: /root/anaconda3/envs/caffe/lib/libboost_date_time.so
tools/finetune_net: /root/anaconda3/envs/caffe/lib/libboost_atomic.so
tools/finetune_net: /root/anaconda3/envs/caffe/lib/libglog.so
tools/finetune_net: /root/anaconda3/envs/caffe/lib/libgflags.so
tools/finetune_net: /root/anaconda3/envs/caffe/lib/libprotobuf.so
tools/finetune_net: /root/anaconda3/envs/caffe/lib/libboost_filesystem.so
tools/finetune_net: /root/anaconda3/envs/caffe/lib/libboost_regex.so
tools/finetune_net: /root/anaconda3/envs/caffe/lib/libboost_chrono.so
tools/finetune_net: /root/anaconda3/envs/caffe/lib/libboost_date_time.so
tools/finetune_net: /root/anaconda3/envs/caffe/lib/libboost_atomic.so
tools/finetune_net: /root/anaconda3/envs/caffe/lib/libglog.so
tools/finetune_net: /root/anaconda3/envs/caffe/lib/libgflags.so
tools/finetune_net: /root/anaconda3/envs/caffe/lib/libprotobuf.so
tools/finetune_net: /root/anaconda3/envs/caffe/lib/libhdf5_cpp.so
tools/finetune_net: /root/anaconda3/envs/caffe/lib/libhdf5.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/librt.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libpthread.so
tools/finetune_net: /root/anaconda3/envs/caffe/lib/libz.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libdl.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libm.so
tools/finetune_net: /root/anaconda3/envs/caffe/lib/libhdf5_hl_cpp.so
tools/finetune_net: /root/anaconda3/envs/caffe/lib/libhdf5_hl.so
tools/finetune_net: /root/anaconda3/envs/caffe/lib/liblmdb.so
tools/finetune_net: /root/anaconda3/envs/caffe/lib/libleveldb.so
tools/finetune_net: /root/anaconda3/envs/caffe/lib/libsnappy.so
tools/finetune_net: /root/anaconda3/envs/caffe/lib/libturbojpeg.so.0
tools/finetune_net: /usr/local/cuda/lib64/libcudart.so
tools/finetune_net: /usr/local/cuda/lib64/libcurand.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libcublas.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libcudnn.so
tools/finetune_net: /root/anaconda3/envs/caffe/lib/libopencv_highgui.so.3.4.2
tools/finetune_net: /root/anaconda3/envs/caffe/lib/libopencv_videoio.so.3.4.2
tools/finetune_net: /root/anaconda3/envs/caffe/lib/libopencv_imgcodecs.so.3.4.2
tools/finetune_net: /root/anaconda3/envs/caffe/lib/libopencv_imgproc.so.3.4.2
tools/finetune_net: /root/anaconda3/envs/caffe/lib/libopencv_core.so.3.4.2
tools/finetune_net: /root/anaconda3/envs/caffe/lib/libopenblas.so
tools/finetune_net: /root/anaconda3/envs/caffe/lib/libboost_python27.so
tools/finetune_net: /root/anaconda3/envs/caffe/lib/libpython2.7.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libnccl.so
tools/finetune_net: /usr/local/cuda/lib64/stubs/libnvidia-ml.so
tools/finetune_net: tools/CMakeFiles/finetune_net.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/caffe-jacinto/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable finetune_net"
	cd /workspace/caffe-jacinto/tools && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/finetune_net.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tools/CMakeFiles/finetune_net.dir/build: tools/finetune_net

.PHONY : tools/CMakeFiles/finetune_net.dir/build

tools/CMakeFiles/finetune_net.dir/clean:
	cd /workspace/caffe-jacinto/tools && $(CMAKE_COMMAND) -P CMakeFiles/finetune_net.dir/cmake_clean.cmake
.PHONY : tools/CMakeFiles/finetune_net.dir/clean

tools/CMakeFiles/finetune_net.dir/depend:
	cd /workspace/caffe-jacinto && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /workspace/caffe-jacinto /workspace/caffe-jacinto/tools /workspace/caffe-jacinto /workspace/caffe-jacinto/tools /workspace/caffe-jacinto/tools/CMakeFiles/finetune_net.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tools/CMakeFiles/finetune_net.dir/depend
