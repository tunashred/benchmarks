#pragma once

#include <gtest/gtest.h>

constexpr size_t CACHE_LINE = 64;

template <typename T>
class CArray : public testing::TestWithParam<size_t> {
protected:
    void SetUp() override;

    void TearDown() override;
public:
    T* array;

    static std::string getTestCaseName(const ::testing::TestParamInfo<size_t>& info) {
        return "size_" + std::to_string(info.param);
    }
};

