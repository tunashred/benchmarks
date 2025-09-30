set(UTILS_LIB "utils")

file(GLOB CXX_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB CXX_HEADERS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/utils/*.hpp)

add_library(${UTILS_LIB} STATIC ${CXX_SOURCES} ${CXX_HEADERS})

target_compile_options(${UTILS_LIB} PRIVATE -mavx
                                            -mavx2)

target_include_directories(${UTILS_LIB} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

include_directories(include)
