set(CMAKE_SYSTEM_NAME "Linux")
set(CMAKE_SYSTEM_PROCESSOR "aarch64")

# specify the cross compiler
set(CMAKE_C_COMPILER "$ENV{CROSS}gcc")
set(CMAKE_CXX_COMPILER "$ENV{CROSS}g++")
# where is the target environment
set(CMAKE_SYSROOT $ENV{CROSS_SYS})
set(CMAKE_FIND_ROOT_PATH "")

set(Boost_INCLUDE_DIR $ENV{BOOST_PATH})

# search for programs in the build host directories (not necessary)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# for libraries and headers in the target directories
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(THREADS_PTHREAD_ARG "2" CACHE STRING "Result from TRY_RUN" FORCE)

set(CMAKE_MODE USE_NEON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAARCH64 ")

set(GNUCC_ARCH "armv8-a+fp+simd+crc+crypto")
set(TUNE_FLAG "cortex-a72")

