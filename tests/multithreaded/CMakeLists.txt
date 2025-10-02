set(MULTITHREADED_SCALAR_LIB "multithreaded_scalar")

file(GLOB CXX_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cache/src/*.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/compute/src/*.cpp)

file(GLOB CXX_HEADERS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cache/include/cache/*.hpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/compute/include/compute/*.hpp)

add_library(${MULTITHREADED_SCALAR_LIB} STATIC ${CXX_SOURCES} ${CXX_HEADERS})

target_include_directories(${MULTITHREADED_SCALAR_LIB} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cache/include
                                                              ${CMAKE_CURRENT_SOURCE_DIR}/compute/include)

target_link_libraries(${MULTITHREADED_SCALAR_LIB} PUBLIC utils)

target_compile_options(${MULTITHREADED_SCALAR_LIB} PUBLIC ${DISABLE_VECTORIZATION_FLAGS})

get_target_property(TARGET_COMPILE_OPTIONS "${MULTITHREADED_SCALAR_LIB}" COMPILE_OPTIONS)

message(STATUS "Compile options for ${MULTITHREADED_SCALAR_LIB}: ${TARGET_COMPILE_OPTIONS}")

foreach(SRC_FILE IN LISTS CXX_SOURCES)
    generate_assembly_for_file(${SRC_FILE} ${EXECUTABLE} ${TARGET_COMPILE_OPTIONS})
endforeach()

add_subdirectory(cache)
add_subdirectory(compute)
