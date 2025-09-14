#pragma once

#include <thread>
#include <gtest/gtest.h>
template <typename T>
class CMatrixArray : public testing::TestWithParam<size_t> {
protected:
    void SetUp() override;

    void TearDown() override;
public:
    T* matrix_A;
    T* matrix_B;
    T* matrix_C;

    std::vector<std::thread> threads;
    
    void naive_mul();
    
    void optimized_mul();

    static std::string getTestCaseName(const ::testing::TestParamInfo<size_t>& info) {
        size_t totalSize = info.param;
        return "size_" + std::to_string(totalSize) + "x" + std::to_string(totalSize);
    }
};
