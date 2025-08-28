#pragma once

#include "utils/utils.hpp"

#include <gtest/gtest.h>
#include <thread>

using testParams = std::tuple<
    std::tuple<size_t, size_t>,
    std::tuple<double, double, double>,
    size_t>;

class AlignedArraySharedComputeMandelbrot : public testing::TestWithParam<testParams> {
protected:
    void SetUp() override;

    void TearDown() override;
public:
    int* array;
    
    std::vector<std::thread> threads;

    size_t width;
    size_t height;

    double top_left_coord_real;
    double top_left_coord_im;
    double pixel_width;

    static std::string getTestCaseName(const ::testing::TestParamInfo<testParams>& info) {
        std::tuple<size_t, size_t> dimensions;
        std::tuple<double, double, double> mandelbrot_args;
        size_t numThreads;
        std::tie(dimensions, mandelbrot_args, numThreads) = info.param;

        size_t width, height;
        double center_coord_real, center_coord_im, radius;
        std::tie(width, height) = dimensions;
        std::tie(center_coord_real, center_coord_im, radius) = mandelbrot_args;

        return "size_" + std::to_string(width) + "x" + std::to_string(height) + "_" + get_mandelbrot_name(radius) + "_numThreads_" + std::to_string(numThreads);
    }

};

static void mandelbrot(size_t start_row, size_t end_row, const AlignedArraySharedComputeMandelbrot* test);

static void runTest(AlignedArraySharedComputeMandelbrot* test);
