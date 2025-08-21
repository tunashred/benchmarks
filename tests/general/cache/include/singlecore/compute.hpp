#pragma once

#include <gtest/gtest.h>

template <typename T>
class CArrayCompute : public testing::TestWithParam<size_t> {
protected:
    void SetUp() override;

    void TearDown() override;
public:
    T* array;

    static std::string getTestCaseName(const ::testing::TestParamInfo<size_t>& info) {
        return "size_" + std::to_string(info.param);
    }

    void batch_add();

    void batch_mul();
};

