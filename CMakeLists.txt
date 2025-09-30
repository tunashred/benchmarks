cmake_minimum_required(VERSION 3.1.0)

project(benchmarks VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#
# Compile flags for any build type
#
set(COMPILE_FLAGS
    -Wall
    -Wextra
    -pthread
    -Wshadow
    -Wcast-align
    -Wwrite-strings
    -Wcast-qual
    -Wconversion
    -Wunreachable-code
    -fstack-usage
    -march=native
    -mtune=native
)

# TODO: Disable vectorization for scalar files
set(RELEASE_FLAGS
    -O3
    -D_FORTIFY_SOURCE=2
)

set(DEBUG_FLAGS
    -g
    -O0
    -fverbose-asm
    -fno-inline
)

set(DISABLE_VECTORIZATION_FLAGS
    -march=x86-64
    -mtune=generic
    -fno-vectorize
    -fno-slp-vectorize
    -mno-sse
    -mno-sse2
    -mno-sse3
    -mno-ssse3
    -mno-sse4.1
    -mno-sse4.2
    -mno-avx
    -mno-avx2
    -mno-avx512f
)

add_compile_options(${COMPILE_FLAGS})

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(${RELEASE_FLAGS})
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(${DEBUG_FLAGS})
endif()

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lstdc++")

# TODO: make this as a separate target
function(generate_assembly_for_file SRC_FILE TARGET_NAME FLAGS)
    get_filename_component(FILE_NAME ${SRC_FILE} NAME)
    get_filename_component(FILE_EXT ${SRC_FILE} EXT)

    if(NOT FILE_EXT MATCHES "\\.(cpp|cc|cxx|c)$")
        return()
    endif()

    file(RELATIVE_PATH REL_PATH ${CMAKE_SOURCE_DIR} ${SRC_FILE})

    string(REPLACE "/" "_" UNIQUE_TARGET_NAME ${REL_PATH})
    string(REPLACE "." "_" UNIQUE_TARGET_NAME ${UNIQUE_TARGET_NAME})
    set(UNIQUE_TARGET_NAME "${UNIQUE_TARGET_NAME}_assembly")

    get_target_property(TARGET_INCLUDE_DIRS ${TARGET_NAME} INCLUDE_DIRECTORIES)
    set(INCLUDE_FLAGS "")
    if(TARGET_INCLUDE_DIRS)
        foreach(INCLUDE_DIR ${TARGET_INCLUDE_DIRS})
            list(APPEND INCLUDE_FLAGS -I${INCLUDE_DIR})
        endforeach()
    endif()

    file(GLOB_RECURSE ALL_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/*/include")
    foreach(INCLUDE_DIR ${ALL_INCLUDE_DIRS})
        list(APPEND INCLUDE_FLAGS -I${INCLUDE_DIR})
    endforeach()
    if(EXISTS "${CMAKE_SOURCE_DIR}/include")
        list(APPEND INCLUDE_FLAGS -I${CMAKE_SOURCE_DIR}/include)
    endif()

    get_filename_component(ABS_SRC_FILE ${SRC_FILE} ABSOLUTE)

    set(ASM_FILE "${CMAKE_CURRENT_BINARY_DIR}/asm/${REL_PATH}.s")

    get_filename_component(ASM_DIR ${ASM_FILE} DIRECTORY)
    file(MAKE_DIRECTORY ${ASM_DIR})

    add_custom_target(${UNIQUE_TARGET_NAME} ALL
        COMMAND ${CMAKE_CXX_COMPILER}
            ${COMPILE_FLAGS}
            ${FLAGS}
            ${INCLUDE_FLAGS}
            -S
            -masm=intel
            -fno-asynchronous-unwind-tables
            -fno-dwarf2-cfi-asm
            -fno-stack-protector
            ${ABS_SRC_FILE}
            -o ${ASM_FILE}
        COMMENT "Generating assembly for ${REL_PATH}"
        VERBATIM
    )
endfunction()

find_package(GTest REQUIRED)
include(GoogleTest)

enable_testing()

add_subdirectory(general)
add_subdirectory(tests)

##################################################################
# Commentary on compile flags
#
# Common flags
# -Wall: Enable all warnings
# -pthread: Enable pthread support
# -Wshadow: Warns if a local variable shadows a function parameter or global variable, or when a lib function is shadowed
# -Wpointer-arith: Warns about questionable pointer operations or pointer arithmetic with void* type or function pointers
# -Wcast-align: Warns for potentially unsafe pointer casts related to strict pointer alignment
# -Wwrite-strings: Warns if a string literal is assigned to a char*
# -Wcast-qual: Warns when casting removes type qualifiers such as const or volatile
# -Wconversion: Warns about implicit type conversions that can lead to data loss
# -Wunreachable-code: Detects code that will never be executed
# -fstack-usage: Generates .su files for each source file which contains stack usage info of each function
# -march=native: Makes use of host CPU specific instructions
# -mtune=native: Optimizes instruction scheduling, prefetching, branch prediction for host CPU
#
# Link flags
# -lm: needed for math.h
# -pg: For gprof
# -lprofiler: For gperftools
#
# Release flags
# -O3: Max optimization level
# -D_FORTIFY_SOURCE=2: Runtime buffer overflow detection
# -funroll-loops: Unrolls loops
#
# Debug flags
# -g: Include debug info
# -O0: No optimizations
# -fverbose-asm: Puts extra comments to generated assembly code to make it more readable
# -finstrument-functions: Inserts hooks for each function in the program
# -fanalyzer: Static analyser at compile time
# -fno-inline: Prevents compiler from inlining any function
