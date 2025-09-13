#pragma once

#include <cstdlib>
#include <cmath>
#include <string>
#include <immintrin.h>

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

inline void scalar_mandelbrot_quadratic(double z_real, double z_im, double c_real, double c_im, double& rez_real, double& rez_im) {
    rez_real = pow(z_real, 2) - pow(z_im, 2) + c_real;
    rez_im = 2 * z_real * z_im + c_im;
}

void simd_mandelbrot_quadratic(const __m256d& z_real, const __m256d& z_im, const __m256d& c_real, const __m256d& c_im,
                               __m256d& rez_real, __m256d& rez_im);

int scalar_diverge(double real, double im, int num_iters);

__m128i simd_diverge(__m256d c_real, __m256d c_im, const __m128i& num_iters);

std::string get_mandelbrot_name(double radius);

void write_pgm(const char* filename, const int* array, size_t width, size_t height, int ITER_1500);
