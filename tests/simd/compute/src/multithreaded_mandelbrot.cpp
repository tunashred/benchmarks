#include "simd/multithreaded_mandelbrot.hpp"
#include "utils/utils.hpp"
#include "utils/constants.hpp"

void AlignedArraySharedMandelbrot::SetUp() {
    std::tuple<size_t, size_t> dimensions;
    std::tuple<double, double, double> mandelbrot_args;
    size_t numThreads;
    std::tie(dimensions, mandelbrot_args, numThreads) = GetParam();

    std::tie(width, height) = dimensions;
    double center_coord_real, center_coord_im, radius;
    std::tie(center_coord_real, center_coord_im, radius) = mandelbrot_args;

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

void AlignedArraySharedMandelbrot::TearDown() {
    free(array);
}

void mandelbrot_simd(size_t start_row, size_t end_row, const AlignedArraySharedMandelbrot* test) {
    double start_im = test->top_left_coord_im - (static_cast<double>(start_row) * test->pixel_width);
    __m256d im_part = _mm256_set1_pd(start_im);
    __m256d pixel_width_vec = _mm256_set1_pd(test->pixel_width);

    for (size_t i = start_row; i < end_row; i++) {
        __m256d real_base = _mm256_set1_pd(test->top_left_coord_real);
        __m256d lane_offsets = _mm256_set_pd(
            3.0 * test->pixel_width, 
            2.0 * test->pixel_width, 
            test->pixel_width, 
            0.0
        );
        __m256d real_part = _mm256_add_pd(real_base, lane_offsets);

        for (size_t j = 0; j + SIMD_DOUBLE_WIDTH <= test->width; j += SIMD_DOUBLE_WIDTH) {
            size_t array_index = test->width * i + j;
            __m128i iter_count = simd_diverge(real_part, im_part, simd_ints_1500);

            _mm_store_si128(reinterpret_cast<__m128i*>(test->array + array_index), iter_count);

            __m256d step = _mm256_set1_pd(SIMD_DOUBLE_WIDTH * test->pixel_width);
            real_part = _mm256_add_pd(real_part, step);
        }
        
        for (size_t j = (test->width / SIMD_DOUBLE_WIDTH) * SIMD_DOUBLE_WIDTH; j < test->width; j++) {
            double real_coord = test->top_left_coord_real + j * test->pixel_width;
            double im_coord = test->top_left_coord_im - i * test->pixel_width;
            int iter_count = scalar_diverge(real_coord, im_coord, ITER_1500);
            test->array[test->width * i + j] = iter_count;
        }
        
        im_part = _mm256_sub_pd(im_part, pixel_width_vec);
    }
}

void runTest(AlignedArraySharedMandelbrot* test) {
    std::tuple<size_t, size_t> dimensions;
    std::tuple<double, double, double> mandelbrot_args;
    size_t numThreads;
    std::tie(dimensions, mandelbrot_args, numThreads) = test->GetParam();

    size_t width, height;
    std::tie(width, height) = dimensions;

    size_t rowsPerThread = height / numThreads;
    size_t remainderRows = height % numThreads;
    size_t currentRow = 0;
    
    for (size_t i = 0; i < numThreads; i++) {
        size_t startRow = currentRow;
        size_t endRow = startRow + rowsPerThread + (i < remainderRows ? 1 : 0);
        
        test->threads.emplace_back(mandelbrot_simd, startRow, endRow, test);
        currentRow = endRow;
    }

    for (auto& thread : test->threads) {
        thread.join();
    }
}

TEST_P(AlignedArraySharedMandelbrot, MandelbrotQuadratic) {
    ::runTest(this);
}

INSTANTIATE_TEST_SUITE_P(
    simd_multithreaded_compute,
    AlignedArraySharedMandelbrot,
    ::testing::Combine(
        ::testing::ValuesIn(picture_dimensions),
        ::testing::ValuesIn(mandelbrot_args),
        ::testing::ValuesIn(NUM_THREADS)),
    AlignedArraySharedMandelbrot::getTestCaseName
);
