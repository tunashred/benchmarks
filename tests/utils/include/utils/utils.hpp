#pragma once

#include <array>
#include <cstdlib>

// TODO: move all the constants to a separate file
constexpr size_t CACHE_LINE = 64;

constexpr size_t ALIGNMENT_32      = 32;
constexpr size_t SIMD_INT_WIDTH    = 8;
constexpr size_t SIMD_LONG_WIDTH   = 4;
constexpr size_t SIMD_DOUBLE_WIDTH = 4;

constexpr size_t LOOP_COUNT_18   = 18;
constexpr size_t LOOP_COUNT_200K = 200'000;
constexpr size_t LOOP_COUNT_400K = 400'000;
constexpr size_t LOOP_COUNT_1M   = 1'000'000;

constexpr std::array<size_t, 3> small_pow2 = {8, 16, 32};

void* safe_malloc(size_t size);

template <typename T>
void create_matrix(T**& matrix, size_t size) {
    size_t rowSize = size * sizeof(T);

    matrix = (T**) safe_malloc(size * sizeof(T*));

    for (size_t i = 0; i < size; i++) {
        matrix[i] = (T*) safe_malloc(rowSize);
        memset(matrix[i], 3, rowSize);
    }
}

void free_matrix(void**& matrix, size_t size);
