file(GLOB_RECURSE CXX_SOURCES CONFIGURE_DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/singlecore/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multithreaded/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp
)

file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/singlecore/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multithreaded/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.hpp
)

set(EXECUTABLE_SCALAR "benchmarks_scalar")

add_executable(${EXECUTABLE_SCALAR} ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp ${CXX_SOURCES} ${HEADERS})

target_compile_options(${EXECUTABLE_SCALAR} PRIVATE ${DISABLE_VECTORIZATION_FLAGS})

target_include_directories(${EXECUTABLE_SCALAR} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/singlecore/cache/include
    ${CMAKE_CURRENT_SOURCE_DIR}/singlecore/compute/include
    ${CMAKE_CURRENT_SOURCE_DIR}/multithreaded/cache/include
    ${CMAKE_CURRENT_SOURCE_DIR}/multithreaded/compute/include
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/include
)
get_target_property(TARGET_COMPILE_OPTIONS "${EXECUTABLE_SCALAR}" COMPILE_OPTIONS)
string(REPLACE ";" " " TARGET_COMPILE_OPTIONS_STR "${TARGET_COMPILE_OPTIONS}")
foreach(SRC_FILE IN LISTS CXX_SOURCES)
    generate_assembly_for_file(${SRC_FILE} ${EXECUTABLE_SCALAR} ${TARGET_COMPILE_OPTIONS_STR})
endforeach()
message(STATUS "Compile options for ${EXECUTABLE_SCALAR}: ${TARGET_COMPILE_OPTIONS}")

# target_link_options(${EXECUTABLE_SCALAR} PRIVATE -lprofiler)

# if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
#     target_link_options(${EXECUTABLE_SCALAR} PRIVATE -fsanitize=address)

#     foreach(SRC_FILE IN LISTS CPP_SOURCES)
#         generate_assembly_for_file(${SRC_FILE} ${EXECUTABLE_SCALAR} ${DEBUG_FLAGS})
#     endforeach()
# elseif(${CMAKE_BUILD_TYPE} STREQUAL "Release")
#     foreach(SRC_FILE IN LISTS CPP_SOURCES)
#         generate_assembly_for_file(${SRC_FILE} ${EXECUTABLE_SCALAR} ${RELEASE_FLAGS})
#     endforeach()
# endif()

#
# GTest
#
target_link_libraries(${EXECUTABLE_SCALAR}
    GTest::gtest
    GTest::gtest_main
    pthread
)

gtest_discover_tests(${EXECUTABLE_SCALAR} DISCOVERY_TIMEOUT 10)

add_subdirectory(singlecore)
add_subdirectory(multithreaded)
add_subdirectory(simd)
add_subdirectory(utils)
