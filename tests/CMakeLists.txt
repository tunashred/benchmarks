file(GLOB_RECURSE CPP_SOURCES CONFIGURE_DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
)

set(EXECUTABLE "benchmarks")

add_executable(${EXECUTABLE} ${CPP_SOURCES} ${HEADERS})

target_compile_options(${EXECUTABLE} PRIVATE ${COMPILE_FLAGS})

target_include_directories(${EXECUTABLE} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/singlecore/cache/include
    ${CMAKE_CURRENT_SOURCE_DIR}/singlecore/compute/include
    ${CMAKE_CURRENT_SOURCE_DIR}/multithreaded/cache/include
    ${CMAKE_CURRENT_SOURCE_DIR}/simd/cache/include
    ${CMAKE_CURRENT_SOURCE_DIR}/simd/compute/include
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/include
)

if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    target_link_options(${EXECUTABLE} PRIVATE -fsanitize=address)

    foreach(SRC_FILE IN LISTS CPP_SOURCES)
        generate_assembly_for_file(${SRC_FILE} ${EXECUTABLE})
    endforeach()
endif()

#
# GTest
#
target_link_libraries(${EXECUTABLE} 
    GTest::gtest 
    GTest::gtest_main 
    pthread
)

gtest_discover_tests(${EXECUTABLE} DISCOVERY_TIMEOUT 10)

add_subdirectory(singlecore)
add_subdirectory(multithreaded)
add_subdirectory(simd)
add_subdirectory(utils)
