file(GLOB_RECURSE CPP_SOURCES CONFIGURE_DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
)

set(EXECUTABLE "benchmarks")

add_executable(${EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

target_include_directories(${EXECUTABLE} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/singlecore/cache/include
    ${CMAKE_CURRENT_SOURCE_DIR}/singlecore/compute/include
    ${CMAKE_CURRENT_SOURCE_DIR}/multithreaded/cache/include
    ${CMAKE_CURRENT_SOURCE_DIR}/multithreaded/compute/include
    ${CMAKE_CURRENT_SOURCE_DIR}/simd/cache/include
    ${CMAKE_CURRENT_SOURCE_DIR}/simd/compute/include
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/include
)

# target_link_options(${EXECUTABLE} PRIVATE -lprofiler)

# if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
#     target_link_options(${EXECUTABLE} PRIVATE -fsanitize=address)

#     foreach(SRC_FILE IN LISTS CPP_SOURCES)
#         generate_assembly_for_file(${SRC_FILE} ${EXECUTABLE} ${DEBUG_FLAGS})
#     endforeach()
# elseif(${CMAKE_BUILD_TYPE} STREQUAL "Release")
#     foreach(SRC_FILE IN LISTS CPP_SOURCES)
#         generate_assembly_for_file(${SRC_FILE} ${EXECUTABLE} ${RELEASE_FLAGS})
#     endforeach()
# endif()

#
# GTest
#
target_link_libraries(${EXECUTABLE}
    singlecore_scalar
    multithreaded_scalar
    simd
    utils
    GTest::gtest 
    GTest::gtest_main 
    pthread
)

gtest_discover_tests(${EXECUTABLE} DISCOVERY_TIMEOUT 10)

add_subdirectory(singlecore)
add_subdirectory(multithreaded)
add_subdirectory(simd)
add_subdirectory(utils)
