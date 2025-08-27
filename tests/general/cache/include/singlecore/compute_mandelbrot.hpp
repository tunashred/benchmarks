#pragma once

#include "utils/utils.hpp"
#include <gtest/gtest.h>

class CArrayComputeMandelbrot : public testing::TestWithParam<std::tuple<std::tuple<size_t, size_t>, std::tuple<double, double, double>>> {
protected:
    void SetUp() override;

    void TearDown() override;
public:
    size_t* array;
    
    size_t width;
    size_t height;

    double top_left_coord_real;
    double top_left_coord_im;
    double pixel_width;

    static std::string getTestCaseName(const ::testing::TestParamInfo<std::tuple<std::tuple<size_t, size_t>, std::tuple<double, double, double>>>& info) {
        std::tuple<size_t, size_t> dimensions;
        std::tuple<double, double, double> mandelbrot_args;
        size_t width, height;
        double center_coord_real, center_coord_im, radius;
        std::tie(dimensions, mandelbrot_args) = info.param;

        std::tie(width, height) = dimensions;
        std::tie(center_coord_real, center_coord_im, radius) = mandelbrot_args;

        return "size_" + std::to_string(width) + "x" + std::to_string(height) + "_" + get_mandelbrot_name(radius);
    }

    void mandelbrot();
};
