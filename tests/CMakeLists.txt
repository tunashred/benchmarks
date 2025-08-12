file(GLOB_RECURSE C_SOURCES CONFIGURE_DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
)

set(EXECUTABLE "benchmarks")

add_executable(${EXECUTABLE} ${C_SOURCES})

target_compile_options(${EXECUTABLE} PRIVATE ${COMPILE_FLAGS})
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    target_link_options(${EXECUTABLE} PRIVATE -fsanitize=address)
endif()

target_include_directories(${EXECUTABLE} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/general/cache/include
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/include
)

#
# GTest
#
target_link_libraries(${EXECUTABLE} 
    GTest::gtest 
    GTest::gtest_main 
    pthread
)

gtest_discover_tests(${EXECUTABLE} DISCOVERY_TIMEOUT 10)

add_subdirectory(general)
add_subdirectory(utils)
