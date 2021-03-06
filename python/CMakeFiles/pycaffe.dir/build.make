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
include python/CMakeFiles/pycaffe.dir/depend.make

# Include the progress variables for this target.
include python/CMakeFiles/pycaffe.dir/progress.make

# Include the compile flags for this target's objects.
include python/CMakeFiles/pycaffe.dir/flags.make

python/CMakeFiles/pycaffe.dir/caffe/_caffe.cpp.o: python/CMakeFiles/pycaffe.dir/flags.make
python/CMakeFiles/pycaffe.dir/caffe/_caffe.cpp.o: python/caffe/_caffe.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/caffe-jacinto/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object python/CMakeFiles/pycaffe.dir/caffe/_caffe.cpp.o"
	cd /workspace/caffe-jacinto/python && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pycaffe.dir/caffe/_caffe.cpp.o -c /workspace/caffe-jacinto/python/caffe/_caffe.cpp

python/CMakeFiles/pycaffe.dir/caffe/_caffe.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pycaffe.dir/caffe/_caffe.cpp.i"
	cd /workspace/caffe-jacinto/python && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/caffe-jacinto/python/caffe/_caffe.cpp > CMakeFiles/pycaffe.dir/caffe/_caffe.cpp.i

python/CMakeFiles/pycaffe.dir/caffe/_caffe.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pycaffe.dir/caffe/_caffe.cpp.s"
	cd /workspace/caffe-jacinto/python && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/caffe-jacinto/python/caffe/_caffe.cpp -o CMakeFiles/pycaffe.dir/caffe/_caffe.cpp.s

# Object files for target pycaffe
pycaffe_OBJECTS = \
"CMakeFiles/pycaffe.dir/caffe/_caffe.cpp.o"

# External object files for target pycaffe
pycaffe_EXTERNAL_OBJECTS =

lib/_caffe.so: python/CMakeFiles/pycaffe.dir/caffe/_caffe.cpp.o
lib/_caffe.so: python/CMakeFiles/pycaffe.dir/build.make
lib/_caffe.so: lib/libcaffe-nv.so.0.17.0
lib/_caffe.so: /root/anaconda3/envs/caffe/lib/libpython2.7.so
lib/_caffe.so: /root/anaconda3/envs/caffe/lib/libboost_system.so
lib/_caffe.so: /root/anaconda3/envs/caffe/lib/libboost_thread.so
lib/_caffe.so: /root/anaconda3/envs/caffe/lib/libboost_filesystem.so
lib/_caffe.so: /root/anaconda3/envs/caffe/lib/libboost_regex.so
lib/_caffe.so: /root/anaconda3/envs/caffe/lib/libboost_chrono.so
lib/_caffe.so: /root/anaconda3/envs/caffe/lib/libboost_date_time.so
lib/_caffe.so: /root/anaconda3/envs/caffe/lib/libboost_atomic.so
lib/_caffe.so: lib/libproto.a
lib/_caffe.so: /root/anaconda3/envs/caffe/lib/libglog.so
lib/_caffe.so: /root/anaconda3/envs/caffe/lib/libgflags.so
lib/_caffe.so: /root/anaconda3/envs/caffe/lib/libprotobuf.so
lib/_caffe.so: /root/anaconda3/envs/caffe/lib/libboost_filesystem.so
lib/_caffe.so: /root/anaconda3/envs/caffe/lib/libboost_regex.so
lib/_caffe.so: /root/anaconda3/envs/caffe/lib/libboost_chrono.so
lib/_caffe.so: /root/anaconda3/envs/caffe/lib/libboost_date_time.so
lib/_caffe.so: /root/anaconda3/envs/caffe/lib/libboost_atomic.so
lib/_caffe.so: /root/anaconda3/envs/caffe/lib/libhdf5_cpp.so
lib/_caffe.so: /root/anaconda3/envs/caffe/lib/libhdf5.so
lib/_caffe.so: /usr/lib/x86_64-linux-gnu/librt.so
lib/_caffe.so: /usr/lib/x86_64-linux-gnu/libpthread.so
lib/_caffe.so: /root/anaconda3/envs/caffe/lib/libz.so
lib/_caffe.so: /usr/lib/x86_64-linux-gnu/libdl.so
lib/_caffe.so: /usr/lib/x86_64-linux-gnu/libm.so
lib/_caffe.so: /root/anaconda3/envs/caffe/lib/libhdf5_hl_cpp.so
lib/_caffe.so: /root/anaconda3/envs/caffe/lib/libhdf5_hl.so
lib/_caffe.so: /root/anaconda3/envs/caffe/lib/liblmdb.so
lib/_caffe.so: /root/anaconda3/envs/caffe/lib/libleveldb.so
lib/_caffe.so: /root/anaconda3/envs/caffe/lib/libsnappy.so
lib/_caffe.so: /root/anaconda3/envs/caffe/lib/libturbojpeg.so.0
lib/_caffe.so: /usr/local/cuda/lib64/libcudart.so
lib/_caffe.so: /usr/local/cuda/lib64/libcurand.so
lib/_caffe.so: /usr/lib/x86_64-linux-gnu/libcublas.so
lib/_caffe.so: /usr/lib/x86_64-linux-gnu/libcudnn.so
lib/_caffe.so: /root/anaconda3/envs/caffe/lib/libopencv_highgui.so.3.4.2
lib/_caffe.so: /root/anaconda3/envs/caffe/lib/libopencv_videoio.so.3.4.2
lib/_caffe.so: /root/anaconda3/envs/caffe/lib/libopencv_imgcodecs.so.3.4.2
lib/_caffe.so: /root/anaconda3/envs/caffe/lib/libopencv_imgproc.so.3.4.2
lib/_caffe.so: /root/anaconda3/envs/caffe/lib/libopencv_core.so.3.4.2
lib/_caffe.so: /root/anaconda3/envs/caffe/lib/libopenblas.so
lib/_caffe.so: /root/anaconda3/envs/caffe/lib/libboost_python27.so
lib/_caffe.so: /root/anaconda3/envs/caffe/lib/libpython2.7.so
lib/_caffe.so: /usr/lib/x86_64-linux-gnu/libnccl.so
lib/_caffe.so: /usr/local/cuda/lib64/stubs/libnvidia-ml.so
lib/_caffe.so: python/CMakeFiles/pycaffe.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/caffe-jacinto/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library ../lib/_caffe.so"
	cd /workspace/caffe-jacinto/python && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pycaffe.dir/link.txt --verbose=$(VERBOSE)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Creating symlink /workspace/caffe-jacinto/python/caffe/_caffe.so -> /workspace/caffe-jacinto/lib/_caffe.so"
	cd /workspace/caffe-jacinto/python && ln -sf /workspace/caffe-jacinto/lib/_caffe.so /workspace/caffe-jacinto/python/caffe/_caffe.so
	cd /workspace/caffe-jacinto/python && /root/anaconda3/envs/caffe/bin/cmake -E make_directory /workspace/caffe-jacinto/python/caffe/proto
	cd /workspace/caffe-jacinto/python && touch /workspace/caffe-jacinto/python/caffe/proto/__init__.py
	cd /workspace/caffe-jacinto/python && cp /workspace/caffe-jacinto/include/caffe/proto/*.py /workspace/caffe-jacinto/python/caffe/proto/

# Rule to build all files generated by this target.
python/CMakeFiles/pycaffe.dir/build: lib/_caffe.so

.PHONY : python/CMakeFiles/pycaffe.dir/build

python/CMakeFiles/pycaffe.dir/clean:
	cd /workspace/caffe-jacinto/python && $(CMAKE_COMMAND) -P CMakeFiles/pycaffe.dir/cmake_clean.cmake
.PHONY : python/CMakeFiles/pycaffe.dir/clean

python/CMakeFiles/pycaffe.dir/depend:
	cd /workspace/caffe-jacinto && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /workspace/caffe-jacinto /workspace/caffe-jacinto/python /workspace/caffe-jacinto /workspace/caffe-jacinto/python /workspace/caffe-jacinto/python/CMakeFiles/pycaffe.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : python/CMakeFiles/pycaffe.dir/depend

