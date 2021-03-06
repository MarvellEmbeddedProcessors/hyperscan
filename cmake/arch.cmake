# detect architecture features
#
# must be called after determining where compiler intrinsics are defined

if (NEON_BUILD)
    set (INTRIN_INC_H "arm_neon.h")
elseif (HAVE_C_X86INTRIN_H)
    set (INTRIN_INC_H "x86intrin.h")
elseif (HAVE_C_INTRIN_H)
    set (INTRIN_INC_H "intrin.h")
else ()
    message (FATAL_ERROR "No intrinsics header found")
endif ()

if (BUILD_AVX512)
    CHECK_C_COMPILER_FLAG(${SKYLAKE_FLAG} HAS_ARCH_SKYLAKE)
    if (NOT HAS_ARCH_SKYLAKE)
        message (FATAL_ERROR "AVX512 not supported by compiler")
    endif ()
endif ()

if (NEON_BUILD)
    set (CMAKE_REQUIRED_FLAGS "${CMAKE_C_FLAGS} ${EXTRA_C_FLAGS} ${ARCH_C_FLAGS}")
elseif (FAT_RUNTIME)
    # test the highest level microarch to make sure everything works
    if (BUILD_AVX512)
        set (CMAKE_REQUIRED_FLAGS "${CMAKE_C_FLAGS} ${EXTRA_C_FLAGS} ${SKYLAKE_FLAG}")
    else ()
        set (CMAKE_REQUIRED_FLAGS "${CMAKE_C_FLAGS} ${EXTRA_C_FLAGS} -march=core-avx2")
    endif ()
else (NOT FAT_RUNTIME)
    # if not fat runtime, then test given cflags
    set (CMAKE_REQUIRED_FLAGS "${CMAKE_C_FLAGS} ${EXTRA_C_FLAGS} ${ARCH_C_FLAGS}")
endif ()

# ensure we have the minimum of SSSE3 - call a SSSE3 intrinsic
CHECK_C_SOURCE_COMPILES("#include <${INTRIN_INC_H}>
int main() {
    __m128i a = _mm_set1_epi8(1);
    (void)_mm_shuffle_epi8(a, a);
}" HAVE_SSSE3)

# now look for AVX2
CHECK_C_SOURCE_COMPILES("#include <${INTRIN_INC_H}>
#if !defined(__AVX2__)
#error no avx2
#endif

int main(){
    __m256i z = _mm256_setzero_si256();
    (void)_mm256_xor_si256(z, z);
}" HAVE_AVX2)

# and now for AVX512
CHECK_C_SOURCE_COMPILES("#include <${INTRIN_INC_H}>
#if !defined(__AVX512BW__)
#error no avx512bw
#endif

int main(){
    __m512i z = _mm512_setzero_si512();
    (void)_mm512_abs_epi8(z);
}" HAVE_AVX512)

if (NEON_BUILD)
    message(STATUS "Building with NEON support")
    add_definitions(-DUSE_NEON -DARCH_64_BIT)
elseif (FAT_RUNTIME)
    if (NOT HAVE_SSSE3)
        message(FATAL_ERROR "SSSE3 support required to build fat runtime")
    endif ()
    if (NOT HAVE_AVX2)
        message(FATAL_ERROR "AVX2 support required to build fat runtime")
    endif ()
    if (BUILD_AVX512 AND NOT HAVE_AVX512)
        message(FATAL_ERROR "AVX512 support requested but not supported")
    endif ()
else (NOT FAT_RUNTIME)
    if (NOT HAVE_AVX2)
        message(STATUS "Building without AVX2 support")
    endif ()
    if (NOT HAVE_AVX512)
        message(STATUS "Building without AVX512 support")
    endif ()
    if (NOT HAVE_SSSE3)
       # message(FATAL_ERROR "A minimum of SSSE3 compiler support is required")
    if (NEON_BUILD)
        message(STATUS "Building with NEON support")
        add_definitions(-DUSE_NEON -DARCH_64_BIT)
    else()
        message(STATUS "Building using SW definitions for intrinsics")
        add_definitions(-DUSE_SCALAR -DARCH_64_BIT)
    endif ()
    endif ()
endif ()

unset (CMAKE_REQUIRED_FLAGS)
unset (INTRIN_INC_H)
