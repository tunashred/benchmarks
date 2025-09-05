#pragma once

#include <gtest/gtest.h>

#include <thread>

// ugly trick
template <typename T>
class AlignedArrayShared;

template <typename T>
using iterate_function = std::function<void(size_t, size_t, AlignedArrayShared<T>*)>;

template <typename T>
class AlignedArrayShared : public testing::TestWithParam<std::tuple<size_t, size_t>> {
protected:
    void SetUp() override;

    void TearDown() override;
public:
    T* array;
    std::vector<std::thread> threads;

    void runTest(iterate_function<T> iterate);

    static std::string getTestCaseName(const ::testing::TestParamInfo<std::tuple<size_t, size_t>>& info) {
        size_t totalSize, numThreads;
        std::tie(totalSize, numThreads) = info.param;
        return "size_" + std::to_string(totalSize) + "_threads_" + std::to_string(numThreads) + "_sliceSize_" + std::to_string(totalSize / numThreads);
    }
};

template <typename T>
static void sequential_iterate(size_t start, size_t end, const AlignedArrayShared<T>* test);

