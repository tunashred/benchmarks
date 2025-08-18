#pragma once

#include <cstdlib>

constexpr size_t ALIGNMENT_32 = 32;
constexpr size_t SIMD_INT_WIDTH = 8;
constexpr size_t SIMD_LONG_WIDTH = 4;
constexpr size_t SIMD_DOUBLE_WIDTH = 4;

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
