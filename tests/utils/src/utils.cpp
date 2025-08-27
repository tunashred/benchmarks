#include <cstdlib>
#include <new>

#include "utils/utils.hpp"

void* safe_malloc(size_t size) {
    void* memory = malloc(size);
    if (memory == nullptr) {
        throw std::bad_alloc();
    }
    return memory;
}

// TODO: actually does free accept void* memory pointers?
void free_matrix(void**& matrix, size_t size) {
    for (size_t i = 0; i < size; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

size_t diverge(double c_real, double c_im, size_t num_iters) {
    size_t i = 0;
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
