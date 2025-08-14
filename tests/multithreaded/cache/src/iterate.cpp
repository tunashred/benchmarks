#include "multithreaded/iterate.hpp"
#include "singlecore/iterate.hpp"
#include "utils/utils.hpp"

template <typename T>
void CArrayShared<T>::SetUp() {
    // TODO: revisit this
    size_t size, _;
    std::tie(size, _) = this->GetParam();
    size_t totalSize = size * sizeof *array;

    array = (T*) safe_malloc(totalSize);
    ASSERT_NE(array, nullptr) << "Unable to alloc array of size " << size;
    // warming up the memory
    memset(array, 1, totalSize);
}

template <typename T>
void CArrayShared<T>::TearDown() {
    free(array);
    threads.clear();
}

template <typename T>
void sequential_iterate(size_t start, size_t end, const CArrayShared<T>* test) {
    for (size_t i = 0; i < LOOP_COUNT; i++) {
        for (size_t j = start; j < end; j++) {
            test->array[j]++;
        }
    }
}

template <typename T>
void reverse_sequential_iterate(size_t start, size_t end, const CArrayShared<T>* test) {
    for (size_t i = 0; i < LOOP_COUNT; i++) {
        for (int j = (int)end - 1; j >= (int)start; j--) {
            test->array[j]++;
        }
    }
}

template <typename T>
void jump_iterate(size_t start, size_t end, const CArrayShared<T>* test) {
    constexpr size_t element_size = sizeof(T),
                     jump_size    = CACHE_LINE / element_size;
    
    for (size_t i = 0; i < LOOP_COUNT; i++) {
        for (size_t k = start; k < jump_size; k++) {
            size_t j = k;
            while (j < end) {
                test->array[j]++;
        
                j += jump_size;
            }
        }
    }
}

template <typename T>
void reverse_jump_iterate(size_t start, size_t end, const CArrayShared<T>* test) {
    constexpr size_t element_size = sizeof(T),
                     jump_size    = CACHE_LINE / element_size;

    for (size_t i = 0; i < LOOP_COUNT; i++) {
        for (size_t k = 0; k < jump_size; k++) {
            size_t remainder = (end - 1 - start - k) % jump_size;
            int j = (int)(end - 1 - remainder);
            while (j >= (int)start) {
                test->array[j]++;
        
                j -= jump_size;
            }
        }
    }
}

template <typename T>
void CArrayShared<T>::runTest(iterate_function<T> iterate, size_t totalSize, size_t numThreads) {
    size_t sliceSize = totalSize / numThreads, 
           remainder = totalSize % numThreads,
           start     = 0;
    
    for (size_t i = 0; i < numThreads; i++) {
        size_t currentSliceSize = sliceSize + (i < remainder ? 1 : 0);
        size_t end = start + currentSliceSize;
        
        threads.emplace_back(iterate, start, end, this);
        start = end;
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

using CArraySharedInt = CArrayShared<int>;
using CArraySharedLong = CArrayShared<long>;
using CArraySharedDouble = CArrayShared<double>;

TEST_P(CArraySharedInt, SequentialIterate) {
    size_t totalSize, numThreads;
    std::tie(totalSize, numThreads) = this->GetParam();
    this->runTest(sequential_iterate<int>, totalSize, numThreads);
}

TEST_P(CArraySharedLong, SequentialIterate) {
    size_t totalSize, numThreads;
    std::tie(totalSize, numThreads) = this->GetParam();
    this->runTest(sequential_iterate<long>, totalSize, numThreads);
}

TEST_P(CArraySharedDouble, SequentialIterate) {
    size_t totalSize, numThreads;
    std::tie(totalSize, numThreads) = this->GetParam();
    this->runTest(sequential_iterate<double>, totalSize, numThreads);
}

TEST_P(CArraySharedInt, ReverseSequentialIterate) {
    size_t totalSize, numThreads;
    std::tie(totalSize, numThreads) = this->GetParam();
    this->runTest(reverse_sequential_iterate<int>, totalSize, numThreads);
}

TEST_P(CArraySharedLong, ReverseSequentialIterate) {
    size_t totalSize, numThreads;
    std::tie(totalSize, numThreads) = this->GetParam();
    this->runTest(reverse_sequential_iterate<long>, totalSize, numThreads);
}

TEST_P(CArraySharedDouble, ReverseSequentialIterate) {
    size_t totalSize, numThreads;
    std::tie(totalSize, numThreads) = this->GetParam();
    this->runTest(reverse_sequential_iterate<double>, totalSize, numThreads);
}

TEST_P(CArraySharedInt, JumpIterate) {
    size_t totalSize, numThreads;
    std::tie(totalSize, numThreads) = this->GetParam();
    this->runTest(jump_iterate<int>, totalSize, numThreads);
}

TEST_P(CArraySharedLong, JumpIterate) {
    size_t totalSize, numThreads;
    std::tie(totalSize, numThreads) = this->GetParam();
    this->runTest(jump_iterate<long>, totalSize, numThreads);
}

TEST_P(CArraySharedDouble, JumpIterate) {
    size_t totalSize, numThreads;
    std::tie(totalSize, numThreads) = this->GetParam();
    this->runTest(jump_iterate<double>, totalSize, numThreads);
}

TEST_P(CArraySharedInt, ReverseJumpIterate) {
    size_t totalSize, numThreads;
    std::tie(totalSize, numThreads) = this->GetParam();
    this->runTest(reverse_jump_iterate<int>, totalSize, numThreads);
}

TEST_P(CArraySharedLong, ReverseJumpIterate) {
    size_t totalSize, numThreads;
    std::tie(totalSize, numThreads) = this->GetParam();
    this->runTest(reverse_jump_iterate<long>, totalSize, numThreads);
}

TEST_P(CArraySharedDouble, ReverseJumpIterate) {
    size_t totalSize, numThreads;
    std::tie(totalSize, numThreads) = this->GetParam();
    this->runTest(reverse_jump_iterate<double>, totalSize, numThreads);
}

std::vector<size_t> sizes = {10, 100, 1000, 10000, 100000, 1000000, 2000000};

std::vector<size_t> threads = {2, 4, 8, 14};

INSTANTIATE_TEST_SUITE_P(
    multithreaded_caching,
    CArraySharedInt,
    ::testing::Combine(
        ::testing::ValuesIn(sizes),
        ::testing::ValuesIn(threads)
    ),
    CArraySharedInt::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
    multithreaded_caching,
    CArraySharedLong,
    ::testing::Combine(
        ::testing::ValuesIn(sizes),
        ::testing::ValuesIn(threads)
    ),
    CArraySharedLong::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
    multithreaded_caching,
    CArraySharedDouble,
    ::testing::Combine(
        ::testing::ValuesIn(sizes),
        ::testing::ValuesIn(threads)
    ),
    CArraySharedDouble::getTestCaseName
);
