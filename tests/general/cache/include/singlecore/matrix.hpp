#pragma once

#include <gtest/gtest.h>

template <typename T>
class CMatrix : public testing::TestWithParam<size_t> {
protected:
    void SetUp() override;

    void TearDown() override;
public:
    T** matrix_A;
    T** matrix_B;
    T** matrix_C;

    void naive_mul();

    void optimized_mul();

    static std::string getTestCaseName(const ::testing::TestParamInfo<size_t>& info) {
        return "size_" + std::to_string(info.param) + "x" + std::to_string(info.param);
    }
};
