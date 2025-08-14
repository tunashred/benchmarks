#pragma once

#include <gtest/gtest.h>

#include <thread>

// ugly trick
template <typename T>
class CArrayShared;

template <typename T>
using iterate_function = std::function<void(size_t, size_t, CArrayShared<T>* test)>;

template <typename T>
class CArrayShared : public testing::TestWithParam<std::tuple<size_t, size_t>> {
protected:
    void SetUp() override;

    void TearDown() override;
public:
    T* array;
    std::vector<std::thread> threads;
    
    void runTest(iterate_function<T> iterate, size_t totalSize, size_t numThreads);

    static std::string getTestCaseName(const ::testing::TestParamInfo<std::tuple<size_t, size_t>>& info) {
        size_t totalSize, numThreads;
        std::tie(totalSize, numThreads) = info.param;
        return "totalSize_" + std::to_string(totalSize) + "_threads_" + std::to_string(numThreads) + "_sliceSize_" + std::to_string(totalSize / numThreads);
    }
};

template <typename T>
static void sequential_iterate(size_t start, size_t end, const CArrayShared<T>* test);

template <typename T>
void reverse_sequential_iterate(size_t start, size_t end, const CArrayShared<T>* test);

template <typename T>
static void jump_iterate(size_t start, size_t end, const CArrayShared<T>* test);

template <typename T>
static void reverse_jump_iterate(size_t start, size_t end, const CArrayShared<T>* test);
