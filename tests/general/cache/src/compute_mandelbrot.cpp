#include "singlecore/compute_mandelbrot.hpp"
#include "utils/utils.hpp"
#include "utils/constants.hpp"

void CArrayComputeMandelbrot::SetUp() {
    size_t width, height;
    std::tie(width, height) = GetParam();

    size_t totalSize = width * height * sizeof *array;

    // TODO: safe_malloc should be wrapped around NO_THROW macro
    array = (int*) safe_malloc(totalSize);
    ASSERT_NE(array, nullptr) << "Unable to alloc array of size " << totalSize;

    // warming up the memory
    memset(array, 1, totalSize);
}

void CArrayComputeMandelbrot::TearDown() {
    free(array);
}

void CArrayComputeMandelbrot::mandelbrot() {
    size_t width, height;
    std::tie(width, height) = GetParam();
    
    double real, im;
    int count = 0;
    
    for (size_t i = 0; i < width; i++) {
        for (size_t j = 0; j < height; j++) {
            for (size_t k = 0; k < LOOP_COUNT_1500; k++) {
                scalar_mandelbrot_quadratic(-0.454839, -0.153992, -0.700025, -0.268500, &real, &im);
                count++;
            }
            array[height * i + j] = count;
        }
    }
}

TEST_P(CArrayComputeMandelbrot, MandelbrotDiverge) {
    mandelbrot();
}

INSTANTIATE_TEST_SUITE_P(
    singlecore_compute,
    CArrayComputeMandelbrot,
    ::testing::ValuesIn(picture_dimensions),
    CArrayComputeMandelbrot::getTestCaseName
);
