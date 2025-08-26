#pragma once

#include <gtest/gtest.h>

class CArrayComputeMandelbrot : public testing::TestWithParam<std::tuple<size_t, size_t>> {
protected:
    void SetUp() override;

    void TearDown() override;
public:
    int* array;

    static std::string getTestCaseName(const ::testing::TestParamInfo<std::tuple<size_t, size_t>>& info) {
        size_t width, height;
        std::tie(width, height) = info.param;
        return "size_" + std::to_string(width) + "x" + std::to_string(height);
    }

    void mandelbrot();
};
