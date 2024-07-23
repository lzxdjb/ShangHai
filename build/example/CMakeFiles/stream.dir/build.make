# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

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
CMAKE_COMMAND = /home/lzx/anaconda3/lib/python3.11/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/lzx/anaconda3/lib/python3.11/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /media/lzx/lzx/ShangHai/Accerlation

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/lzx/lzx/ShangHai/Accerlation/build

# Include any dependencies generated for this target.
include example/CMakeFiles/stream.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include example/CMakeFiles/stream.dir/compiler_depend.make

# Include the progress variables for this target.
include example/CMakeFiles/stream.dir/progress.make

# Include the compile flags for this target's objects.
include example/CMakeFiles/stream.dir/flags.make

example/CMakeFiles/stream.dir/stream.cu.o: example/CMakeFiles/stream.dir/flags.make
example/CMakeFiles/stream.dir/stream.cu.o: example/CMakeFiles/stream.dir/includes_CUDA.rsp
example/CMakeFiles/stream.dir/stream.cu.o: /media/lzx/lzx/ShangHai/Accerlation/example/stream.cu
example/CMakeFiles/stream.dir/stream.cu.o: example/CMakeFiles/stream.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/lzx/lzx/ShangHai/Accerlation/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object example/CMakeFiles/stream.dir/stream.cu.o"
	cd /media/lzx/lzx/ShangHai/Accerlation/build/example && /usr/local/cuda-11.8/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT example/CMakeFiles/stream.dir/stream.cu.o -MF CMakeFiles/stream.dir/stream.cu.o.d -x cu -rdc=true -c /media/lzx/lzx/ShangHai/Accerlation/example/stream.cu -o CMakeFiles/stream.dir/stream.cu.o

example/CMakeFiles/stream.dir/stream.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/stream.dir/stream.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

example/CMakeFiles/stream.dir/stream.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/stream.dir/stream.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target stream
stream_OBJECTS = \
"CMakeFiles/stream.dir/stream.cu.o"

# External object files for target stream
stream_EXTERNAL_OBJECTS =

example/CMakeFiles/stream.dir/cmake_device_link.o: example/CMakeFiles/stream.dir/stream.cu.o
example/CMakeFiles/stream.dir/cmake_device_link.o: example/CMakeFiles/stream.dir/build.make
example/CMakeFiles/stream.dir/cmake_device_link.o: src/libtinympc.a
example/CMakeFiles/stream.dir/cmake_device_link.o: example/CMakeFiles/stream.dir/deviceLinkLibs.rsp
example/CMakeFiles/stream.dir/cmake_device_link.o: example/CMakeFiles/stream.dir/deviceObjects1
example/CMakeFiles/stream.dir/cmake_device_link.o: example/CMakeFiles/stream.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/lzx/lzx/ShangHai/Accerlation/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/stream.dir/cmake_device_link.o"
	cd /media/lzx/lzx/ShangHai/Accerlation/build/example && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/stream.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
example/CMakeFiles/stream.dir/build: example/CMakeFiles/stream.dir/cmake_device_link.o
.PHONY : example/CMakeFiles/stream.dir/build

# Object files for target stream
stream_OBJECTS = \
"CMakeFiles/stream.dir/stream.cu.o"

# External object files for target stream
stream_EXTERNAL_OBJECTS =

example/stream: example/CMakeFiles/stream.dir/stream.cu.o
example/stream: example/CMakeFiles/stream.dir/build.make
example/stream: src/libtinympc.a
example/stream: example/CMakeFiles/stream.dir/cmake_device_link.o
example/stream: example/CMakeFiles/stream.dir/linkLibs.rsp
example/stream: example/CMakeFiles/stream.dir/objects1
example/stream: example/CMakeFiles/stream.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/lzx/lzx/ShangHai/Accerlation/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable stream"
	cd /media/lzx/lzx/ShangHai/Accerlation/build/example && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/stream.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
example/CMakeFiles/stream.dir/build: example/stream
.PHONY : example/CMakeFiles/stream.dir/build

example/CMakeFiles/stream.dir/clean:
	cd /media/lzx/lzx/ShangHai/Accerlation/build/example && $(CMAKE_COMMAND) -P CMakeFiles/stream.dir/cmake_clean.cmake
.PHONY : example/CMakeFiles/stream.dir/clean

example/CMakeFiles/stream.dir/depend:
	cd /media/lzx/lzx/ShangHai/Accerlation/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/lzx/lzx/ShangHai/Accerlation /media/lzx/lzx/ShangHai/Accerlation/example /media/lzx/lzx/ShangHai/Accerlation/build /media/lzx/lzx/ShangHai/Accerlation/build/example /media/lzx/lzx/ShangHai/Accerlation/build/example/CMakeFiles/stream.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : example/CMakeFiles/stream.dir/depend
