#pragma once

#include <gtest/gtest.h>
#include <thread>

template <typename T>
class AlignedMatrixShared;

template <typename T>
using mul_function = void(*)(size_t, size_t, size_t, size_t, const AlignedMatrixShared<T>*);

template <typename T>
class AlignedMatrixShared : public testing::TestWithParam<std::tuple<size_t, size_t>> {
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

    void runTest(mul_function<T> mul);

    static std::string getTestCaseName(const ::testing::TestParamInfo<std::tuple<size_t, size_t>>& info) {
        size_t totalSize, numThreads;
        std::tie(totalSize, numThreads) = info.param;
        return "size_" + std::to_string(totalSize) + "x" + std::to_string(totalSize) + "_threads_" + std::to_string(numThreads);
    }
};

template <typename T>
void optimized_mul(size_t startRow, size_t endRow, size_t startCol, size_t endCol, const AlignedMatrixShared<T>* test);
