#pragma once

#include <gtest/gtest.h>

template <typename T>
class CArray : public testing::TestWithParam<size_t> {
protected:
    void SetUp() override;

    void TearDown() override;
public:
    T* array;
};

template <typename T>
void iterate(const CArray<T>* test);

std::string getTestCaseName(const ::testing::TestParamInfo<size_t>& info) {
    return "size_" + std::to_string(info.param);
}
