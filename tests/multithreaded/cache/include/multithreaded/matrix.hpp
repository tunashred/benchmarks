#pragma once

#include <thread>
#include <gtest/gtest.h>

// ugly trick
template <typename T>
class CMatrixShared;

template <typename T>
using mul_function = std::function<void(size_t, size_t, size_t, size_t, CMatrixShared<T>*)>;

template <typename T>
class CMatrixShared : public testing::TestWithParam<std::tuple<size_t, size_t>> {
protected:
    void SetUp() override;

    void TearDown() override;
public:
    T** matrix_A;
    T** matrix_B;
    T** matrix_C;

    std::vector<std::thread> threads;
    
    void runTest(mul_function<T> mul);
    
    static std::string getTestCaseName(const ::testing::TestParamInfo<std::tuple<size_t, size_t>>& info) {
        size_t totalSize, numThreads;
        std::tie(totalSize, numThreads) = info.param;
        return "totalSize_" + std::to_string(totalSize) + "x" + std::to_string(totalSize) + "_threads_" + std::to_string(numThreads);
    }
};

template <typename T>
static void naive_mul(size_t startRow, size_t endRow, size_t startCol, size_t endCol, const CMatrixShared<T>* test);

template <typename T>
static void optimized_mul(size_t startRow, size_t endRow, size_t startCol, size_t endCol, const CMatrixShared<T>* test);
