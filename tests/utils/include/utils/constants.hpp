#pragma once

#include <array>
#include <cstddef>
#include <tuple>
#include <immintrin.h>

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

constexpr std::array<std::tuple<size_t, size_t>, 4> picture_dimensions = {{
    {1920, 1080},
    {3840, 2160},
    {5760, 3240},
    {7680, 4320}
}};

constexpr std::array<std::tuple<double, double, double>, 5> mandelbrot_args = {{
    //   center_x             center_y             scale
    { -0.43,                  -0.1,                1.0           }, // mandelbrot
    { -0.75 + 0.00005,        -0.02,               0.00025       }, // shells
    { -0.72413,                0.28644,            0.0004        }, // seastar
    { -0.7,                   -0.26,               0.01          }, // stuff
    { -0.700025 + 0.000000007, -0.26849991525,     0.0000000035  }  // galaxy
}};

constexpr int ITER_1500 = 1500;

const __m256d simd_doubles_2 = _mm256_set1_pd(2);

const __m256d simd_doubles_minus_2 = _mm256_set1_pd(-2);

const __m128i simd_ints_1500 = _mm_set1_epi32(1500);

const __m256d simd_doubles_true = _mm256_castsi256_pd(_mm256_set1_epi64x(-1));

const __m128i simd_ints_1 = _mm_set1_epi32(1);
