#pragma once

#include <array>
#include <cstddef>
#include <tuple>

constexpr size_t CACHE_LINE = 64;

constexpr size_t ALIGNMENT_32      = 32;
constexpr size_t SIMD_INT_WIDTH    = 8;
constexpr size_t SIMD_LONG_WIDTH   = 4;
constexpr size_t SIMD_DOUBLE_WIDTH = 4;

constexpr size_t LOOP_COUNT_18   = 18;
constexpr size_t LOOP_COUNT_1500 = 1500;
constexpr size_t LOOP_COUNT_200K = 200'000;
constexpr size_t LOOP_COUNT_400K = 400'000;
constexpr size_t LOOP_COUNT_1M   = 1'000'000;

constexpr std::array<size_t, 3> small_pow2 = {8, 16, 32};

constexpr std::array<std::tuple<size_t, size_t>, 4> picture_dimensions = {{ {1920, 1080},
                                                                            {3840, 2160},
                                                                            {5760, 3240},
                                                                            {7680, 4320} }};
