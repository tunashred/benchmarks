#pragma once

#include <cstdlib>
#include <cmath>

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

inline void scalar_mandelbrot_quadratic(double z_real, double z_im, double c_real, double c_im, double *rez_real, double *rez_im) {
    *rez_real = pow(z_real, 2) - pow(z_im, 2) + c_real;
    *rez_im = 2 * z_real * z_im + c_im;
}
