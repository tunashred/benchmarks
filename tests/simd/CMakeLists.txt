set(EXECUTABLE_SIMD "benchmarks_simd")

file(GLOB CXX_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cache/src/*.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/compute/src/*.cpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../utils/src/*.cpp)

file(GLOB CXX_HEADERS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cache/include/cache/*.hpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/compute/include/compute/*.hpp
                                        ${CMAKE_CURRENT_SOURCE_DIR}/../utils/*.hpp)

add_executable(${EXECUTABLE_SIMD} ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp ${CXX_SOURCES})

target_compile_options(${EXECUTABLE_SIMD} PRIVATE 
    -mavx
    -mavx2
    -march=native
    -mtune=native
)

target_include_directories(${EXECUTABLE_SIMD} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cache/include
                                                              ${CMAKE_CURRENT_SOURCE_DIR}/compute/include
                                                              ${CMAKE_CURRENT_SOURCE_DIR}/../utils/include)

target_link_libraries(${EXECUTABLE_SIMD} PRIVATE
    GTest::gtest 
    GTest::gtest_main
)

message(STATUS "Utils sources found: ${CXX_SOURCES}")

get_target_property(TARGET_COMPILE_OPTIONS "${EXECUTABLE_SIMD}" COMPILE_OPTIONS)

message(STATUS "Compile options for ${EXECUTABLE_SIMD}: ${TARGET_COMPILE_OPTIONS}")

# foreach(SRC_FILE IN LISTS CXX_SOURCES)
#     generate_assembly_for_file(${SRC_FILE} ${EXECUTABLE} ${TARGET_COMPILE_OPTIONS})
# endforeach()

add_subdirectory(cache)
add_subdirectory(compute)
