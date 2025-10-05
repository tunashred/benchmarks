#include "utils/utils.hpp"
#include "utils/constants.hpp"

#include <cstdint>
#include <cstdlib>
#include <new>
#include <fstream>
#include <stdexcept>
#ifdef __AVX__
#include <immintrin.h>
#endif

void* safe_malloc(size_t size) {
    void* memory = malloc(size);
    if (memory == nullptr) {
        throw std::bad_alloc();
    }
    return memory;
}

void free_matrix(void**& matrix, size_t size) {
    for (size_t i = 0; i < size; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

int scalar_diverge(double c_real, double c_im, int num_iters) {
    int i = 0;
    double z_real = 0, z_im = 0;
    double z_real_returnat, z_im_returnat;
    while( z_real <= 2.f && z_real >= -2.f && i++ <= num_iters ) {
        scalar_mandelbrot_quadratic(z_real, z_im, c_real, c_im, z_real_returnat, z_im_returnat);
        z_real = z_real_returnat;
        z_im = z_im_returnat;
    }
    if(i >= num_iters) {
        return 0;
    }
    return i;
}

#ifdef __AVX__
void simd_mandelbrot_quadratic(const __m256d& z_real, const __m256d& z_im, const __m256d& c_real, const __m256d& c_im,
                               __m256d& rez_real, __m256d& rez_im) {
    // rez_real = z_real * z_real - z_im * z_im + c_real;
    rez_real = _mm256_add_pd(_mm256_sub_pd(_mm256_mul_pd(z_real, z_real), _mm256_mul_pd(z_im, z_im)), c_real);
    // rez_im = 2 * z_real * z_im + c_im;
    rez_im = _mm256_add_pd(_mm256_mul_pd(_mm256_mul_pd(z_real, _mm256_set1_pd(2)), z_im), c_im);
}

__m128i simd_diverge(__m256d c_real, __m256d c_im, const __m128i& num_iters) {
    __m128i i = _mm_set1_epi32(0);
    __m256d z_real = _mm256_set1_pd(0), z_im = _mm256_set1_pd(0);
    __m256d z_real_ret, z_im_ret, mask = simd_doubles_true;

    while(_mm256_movemask_pd(mask) != 0) {
        simd_mandelbrot_quadratic(z_real, z_im, c_real, c_im, z_real_ret, z_im_ret);

        i = _mm_add_epi32(i, simd_ints_1);

        __m256d not_diverged = _mm256_cmp_pd(z_real, simd_doubles_2, _CMP_LE_OQ);
        not_diverged = _mm256_and_pd(not_diverged, _mm256_cmp_pd(z_real, simd_doubles_minus_2, _CMP_GE_OQ));

        __m256d not_max_iter = _mm256_cvtepi32_pd(_mm_cmpgt_epi32(num_iters, i));

        mask = _mm256_and_pd(not_diverged, not_max_iter);

        z_real = _mm256_blendv_pd(z_real, z_real_ret, mask);
        z_im = _mm256_blendv_pd(z_im, z_im_ret, mask);
    }

    return i;
}
#endif

void write_pgm(const char* filename, const int* array, size_t width, size_t height, int iter_count) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Cannot open file for writing");
    }

    ofs << "P5\n" << width << " " << height << "\n255\n";

    for (size_t i = 0; i < width * height; ++i) {
        uint8_t pixel = static_cast<uint8_t>(255 * std::min(array[i], iter_count) / iter_count);
        ofs.write(reinterpret_cast<char*>(&pixel), 1);
    }

    ofs.close();
}

std::string get_mandelbrot_name(double radius) {
    if (radius == 1.0) {
        return "mandelbrot";
    }
    if (radius == 0.00025) {
        return "shells";
    }
    if (radius == 0.0004) {
        return "seastar";
    }
    if (radius == 0.01) {
        return "stuff";
    }
    if (radius == 0.0000000035) {
        return "galaxy";
    }
    return "unknown_mandelbrot_name";
}
