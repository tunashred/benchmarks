#pragma once

#include <gtest/gtest.h>

constexpr size_t CACHE_LINE = 64;
constexpr size_t LOOP_COUNT = 200000;

constexpr size_t ALIGNMENT_32 = 32;
constexpr size_t SIMD_INT_WIDTH = 8;
constexpr size_t SIMD_LONG_WIDTH = 4;
constexpr size_t SIMD_DOUBLE_WIDTH = 4;

template <typename T>
class AlignedArray : public testing::TestWithParam<size_t> {
protected:
    void SetUp() override;

    void TearDown() override;
public:
    T* array;

    static std::string getTestCaseName(const ::testing::TestParamInfo<size_t>& info) {
        return "size_" + std::to_string(info.param);
    }
};
