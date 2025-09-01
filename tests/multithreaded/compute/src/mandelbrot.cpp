#include "multithreaded/mandelbrot.hpp"
#include "utils/utils.hpp"
#include "utils/constants.hpp"

void CArrayShared::SetUp() {
    std::tuple<size_t, size_t> dimensions;
    std::tuple<double, double, double> mandelbrot_args;
    size_t numThreads;
    std::tie(dimensions, mandelbrot_args, numThreads) = GetParam();

    double center_coord_real, center_coord_im, radius;
    std::tie(width, height) = dimensions;
    
    size_t totalSize = width * height * sizeof(int);

    array = (int*) safe_malloc(totalSize);
    ASSERT_NE(array, nullptr) << "Unable to alloc array of size " << totalSize;
    
    // warming up the memory
    memset(array, 1, totalSize);
    
    std::tie(center_coord_real, center_coord_im, radius) = mandelbrot_args;
    this->pixel_width         = radius * 2 / ((width < height) ? static_cast<double>(width) : static_cast<double>(height));
    this->top_left_coord_real = center_coord_real - static_cast<double>(width) / 2 * this->pixel_width;
    this->top_left_coord_im   = center_coord_im + static_cast<double>(height) / 2 * this->pixel_width;
}

void CArrayShared::TearDown() {
    free(array);
}

void mandelbrot(size_t start_row, size_t end_row, const CArrayShared* test) {
    double im_part = test->top_left_coord_im - (static_cast<double>(start_row) * test->pixel_width);
    
    for (size_t i = start_row; i < end_row; i++) {
        double real_part = test->top_left_coord_real;
        for (size_t j = 0; j < test->width; j++) {
            int iter_count = scalar_diverge(real_part, im_part, ITER_1500);
            
            test->array[test->width * i + j] = iter_count;

            real_part += test->pixel_width;
        }
        im_part -= test->pixel_width;
    }
}

void runTest(CArrayShared* test) {
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
        
        test->threads.emplace_back(mandelbrot, startRow, endRow, test);
        currentRow = endRow;
    }

    for (auto& thread : test->threads) {
        thread.join();
    }
}

TEST_P(CArrayShared, MandelbrotQuadratic) {
    ::runTest(this);
}

INSTANTIATE_TEST_SUITE_P(
    multithreaded_compute,
    CArrayShared,
    ::testing::Combine(
        ::testing::ValuesIn(picture_dimensions),
        ::testing::ValuesIn(mandelbrot_args),
        ::testing::ValuesIn(NUM_THREADS)),
    CArrayShared::getTestCaseName
);
