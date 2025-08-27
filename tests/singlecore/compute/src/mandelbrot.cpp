#include "singlecore/mandelbrot.hpp"
#include "utils/utils.hpp"
#include "utils/constants.hpp"

void CArrayComputeMandelbrot::SetUp() {
    std::tuple<size_t, size_t> dimensions;
    std::tuple<double, double, double> _mandelbrot_args;
    double center_coord_real, center_coord_im, radius;
    std::tie(dimensions, _mandelbrot_args) = GetParam();

    std::tie(width, height) = dimensions;
    std::tie(center_coord_real, center_coord_im, radius) = _mandelbrot_args;

    size_t totalSize = width * height * sizeof *array;

    // TODO: safe_malloc should be wrapped around NO_THROW macro
    array = (size_t*) safe_malloc(totalSize);
    ASSERT_NE(array, nullptr) << "Unable to alloc array of size " << totalSize;

    // warming up the memory
    memset(array, 1, totalSize);

    this->pixel_width         = radius * 2 / ((width < height) ? static_cast<double>(width) : static_cast<double>(height));
    this->top_left_coord_real = center_coord_real + static_cast<double>(width) / 2 * this->pixel_width;
    this->top_left_coord_im   = center_coord_im + static_cast<double>(height) / 2 * this->pixel_width;
}

void CArrayComputeMandelbrot::TearDown() {
    free(array);
}

void CArrayComputeMandelbrot::mandelbrot() {
    double im_part = this->top_left_coord_im;
    for (size_t i = 0; i < height; i++) {
        double real_part = this->top_left_coord_real;
        for (size_t j = 0; j < width; j++) {
            size_t iter_count = diverge(real_part, im_part, LOOP_COUNT_1500);
            
            array[height * i + j] = iter_count;

            real_part += pixel_width;
        }
        im_part -= pixel_width;
    }
}

TEST_P(CArrayComputeMandelbrot, MandelbrotQuadratic) {
    mandelbrot();
}

INSTANTIATE_TEST_SUITE_P(
    singlecore_compute,
    CArrayComputeMandelbrot,
    ::testing::Combine(
        ::testing::ValuesIn(picture_dimensions),
        ::testing::ValuesIn(mandelbrot_args)),
    CArrayComputeMandelbrot::getTestCaseName
);
