set(SINGLECORE_SCALAR_LIB "singlecore_scalar")

file(GLOB CXX_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cache/src/*.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/compute/src/*.cpp)

file(GLOB CXX_HEADERS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cache/include/cache/*.hpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/compute/include/compute/*.hpp)

add_library(${SINGLECORE_SCALAR_LIB} STATIC ${CXX_SOURCES} ${CXX_HEADERS})

target_include_directories(${SINGLECORE_SCALAR_LIB} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cache/include
                                                              ${CMAKE_CURRENT_SOURCE_DIR}/compute/include)

target_link_libraries(${SINGLECORE_SCALAR_LIB} PUBLIC utils)

target_compile_options(${SINGLECORE_SCALAR_LIB} PUBLIC ${DISABLE_VECTORIZATION_FLAGS})

foreach(SRC_FILE IN LISTS CXX_SOURCES)
    generate_assembly_for_file(${SRC_FILE} ${EXECUTABLE} ${RELEASE_FLAGS} ${DISABLE_VECTORIZATION_FLAGS})
endforeach()

add_subdirectory(cache)
add_subdirectory(compute)
