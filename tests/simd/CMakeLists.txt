set(SIMD "simd")

file(GLOB CXX_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cache/src/*.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/compute/src/*.cpp)

file(GLOB CXX_HEADERS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cache/include/cache/*.hpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/compute/include/compute/*.hpp)

add_library(${SIMD} STATIC ${CXX_SOURCES} ${CXX_HEADERS})

target_compile_options(${SIMD} PRIVATE -mavx
                                       -mavx2)

target_include_directories(${SIMD} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cache/include
                                                              ${CMAKE_CURRENT_SOURCE_DIR}/compute/include)

target_link_libraries(${SIMD} PUBLIC utils)

add_subdirectory(cache)
add_subdirectory(compute)
