set(MULTITHREADED_SCALAR_LIB "multithreaded_scalar")

file(GLOB CXX_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cache/src/*.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/compute/src/*.cpp)

file(GLOB CXX_HEADERS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cache/include/cache/*.hpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/compute/include/compute/*.hpp)

add_library(${MULTITHREADED_SCALAR_LIB} STATIC ${CXX_SOURCES} ${CXX_HEADERS})

target_include_directories(${MULTITHREADED_SCALAR_LIB} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cache/include
                                                              ${CMAKE_CURRENT_SOURCE_DIR}/compute/include)

target_link_libraries(${MULTITHREADED_SCALAR_LIB} PUBLIC utils)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${MULTITHREADED_SCALAR_LIB} PRIVATE
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
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(
        -fno-tree-vectorize
        -fno-slp-vectorize
        -mno-sse
        -mno-sse2
        -mno-avx
        -mno-avx2
    )
endif()

add_subdirectory(cache)
add_subdirectory(compute)
