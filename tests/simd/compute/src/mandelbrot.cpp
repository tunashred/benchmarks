#include "simd/mandelbrot.hpp"
#include "utils/utils.hpp"
#include "utils/constants.hpp"

void AlignedArrayMandelbrot::SetUp() {
    std::tuple<size_t, size_t> dimensions;
    std::tuple<double, double, double> _mandelbrot_args;
    double center_coord_real, center_coord_im, radius;
    std::tie(dimensions, _mandelbrot_args) = GetParam();

    std::tie(width, height) = dimensions;
    std::tie(center_coord_real, center_coord_im, radius) = _mandelbrot_args;

    size_t totalSize = width * height * sizeof *array;

    if (totalSize % ALIGNMENT_32 != 0) {
        totalSize += ALIGNMENT_32 - (totalSize % ALIGNMENT_32);
    }
    array = static_cast<int*>(std::aligned_alloc(ALIGNMENT_32, totalSize));
    ASSERT_NE(array, nullptr) << "Unable to alloc array of size " << totalSize;

    // warming up the memory
    memset(array, 1, totalSize);

    this->pixel_width         = radius * 2 / ((width < height) ? static_cast<double>(width) : static_cast<double>(height));
    this->top_left_coord_real = center_coord_real - static_cast<double>(width) / 2 * this->pixel_width;
    this->top_left_coord_im   = center_coord_im + static_cast<double>(height) / 2 * this->pixel_width;
}

void AlignedArrayMandelbrot::TearDown() {
    free(array);
}

void AlignedArrayMandelbrot::mandelbrot() {
    __m256d im_part = _mm256_set1_pd(this->top_left_coord_im);
    __m256d pixel_width_vec = _mm256_set1_pd(this->pixel_width);

    for (size_t i = 0; i < height; i++) {
        __m256d real_base = _mm256_set1_pd(this->top_left_coord_real);
        __m256d lane_offets = _mm256_set_pd(3 * this->pixel_width, 2 * this->pixel_width, this->pixel_width, 0);
        __m256d real_part = _mm256_add_pd(real_base, lane_offets);

        for (size_t j = 0; j + SIMD_DOUBLE_WIDTH <= width; j += SIMD_DOUBLE_WIDTH) {
            size_t array_index = width * i + j;
            __m128i iter_count = simd_diverge(real_part, im_part, simd_ints_1500);

            _mm_store_si128(reinterpret_cast<__m128i*>(array + array_index), iter_count);

            __m256d step = _mm256_set1_pd(SIMD_DOUBLE_WIDTH * this->pixel_width);
            real_part = _mm256_add_pd(real_part, step);
        }
        im_part = _mm256_sub_pd(im_part, pixel_width_vec);
    }
}

TEST_P(AlignedArrayMandelbrot, MandelbrotQuadratic) {
    mandelbrot();
}

INSTANTIATE_TEST_SUITE_P(
    simd_singlecore_compute,
    AlignedArrayMandelbrot,
    ::testing::Combine(
        ::testing::ValuesIn(picture_dimensions),
        ::testing::ValuesIn(mandelbrot_args)),
    AlignedArrayMandelbrot::getTestCaseName
);
