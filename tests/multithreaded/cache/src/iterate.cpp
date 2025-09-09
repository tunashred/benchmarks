#include "multithreaded/iterate.hpp"
#include "utils/utils.hpp"
#include "utils/constants.hpp"

template <typename T>
void CArrayShared<T>::SetUp() {
    size_t size;
    std::tie(size, std::ignore) = this->GetParam();
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
    for (size_t i = 0; i < LOOP_COUNT_200K; i++) {
        for (size_t j = start; j < end; j++) {
            test->array[j]++;
        }
    }
}

template <typename T>
void reverse_sequential_iterate(size_t start, size_t end, const CArrayShared<T>* test) {
    for (size_t i = 0; i < LOOP_COUNT_200K; i++) {
        for (int j = (int)end - 1; j >= (int)start; j--) {
            test->array[j]++;
        }
    }
}

template <typename T>
void neighbour_sequential_iterate(size_t size, size_t start, size_t increment, const CArrayShared<T>* test) {
    for (size_t i = 0; i < LOOP_COUNT_200K; i++) {
        for (size_t j = start; j + start < size - 1; j += increment) {
            test->array[j]++;
        }
    }
}

template <typename T>
void jump_iterate(size_t start, size_t end, const CArrayShared<T>* test) {
    constexpr size_t element_size = sizeof(T),
                     jump_size    = CACHE_LINE / element_size;
    
    for (size_t i = 0; i < LOOP_COUNT_200K; i++) {
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

    for (size_t i = 0; i < LOOP_COUNT_200K; i++) {
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
void CArrayShared<T>::runTest(iterate_function<T> iterate) {
    size_t totalSize, numThreads;
    std::tie(totalSize, numThreads) = this->GetParam();

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

template <typename T>
void CArrayShared<T>::runNeighbourTest() {
    size_t totalSize, numThreads;
    std::tie(totalSize, numThreads) = this->GetParam();
    
    for (size_t i = 0; i < numThreads; i++) {
        threads.emplace_back(neighbour_sequential_iterate<T>, totalSize, i, numThreads, this);
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

using CArraySharedInt = CArrayShared<int>;
using CArraySharedLong = CArrayShared<long>;
using CArraySharedDouble = CArrayShared<double>;

TEST_P(CArraySharedInt, SequentialIterate) {
    this->runTest(sequential_iterate<int>);
}

TEST_P(CArraySharedLong, SequentialIterate) {
    this->runTest(sequential_iterate<long>);
}

TEST_P(CArraySharedDouble, SequentialIterate) {
    this->runTest(sequential_iterate<double>);
}

TEST_P(CArraySharedInt, DISABLED_ReverseSequentialIterate) {
    this->runTest(reverse_sequential_iterate<int>);
}

TEST_P(CArraySharedLong, DISABLED_ReverseSequentialIterate) {
    this->runTest(reverse_sequential_iterate<long>);
}

TEST_P(CArraySharedDouble, DISABLED_ReverseSequentialIterate) {
    this->runTest(reverse_sequential_iterate<double>);
}

TEST_P(CArraySharedInt, DISABLED_NeighbourSequentialIterate) {
    this->runNeighbourTest();
}

TEST_P(CArraySharedLong, DISABLED_NeighbourSequentialIterate) {
    this->runNeighbourTest();
}

TEST_P(CArraySharedDouble, DISABLED_NeighbourSequentialIterate) {
    this->runNeighbourTest();
}

TEST_P(CArraySharedInt, JumpIterate) {
    this->runTest(jump_iterate<int>);
}

TEST_P(CArraySharedLong, JumpIterate) {
    this->runTest(jump_iterate<long>);
}

TEST_P(CArraySharedDouble, JumpIterate) {
    this->runTest(jump_iterate<double>);
}

TEST_P(CArraySharedInt, ReverseJumpIterate) {
    this->runTest(reverse_jump_iterate<int>);
}

TEST_P(CArraySharedLong, ReverseJumpIterate) {
    this->runTest(reverse_jump_iterate<long>);
}

TEST_P(CArraySharedDouble, ReverseJumpIterate) {
    this->runTest(reverse_jump_iterate<double>);
}

INSTANTIATE_TEST_SUITE_P(
    multithreaded_caching,
    CArraySharedInt,
    ::testing::Combine(
        ::testing::ValuesIn(ARRAY_SIZES),
        ::testing::ValuesIn(NUM_THREADS)
    ),
    CArraySharedInt::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
    multithreaded_caching,
    CArraySharedLong,
    ::testing::Combine(
        ::testing::ValuesIn(ARRAY_SIZES),
        ::testing::ValuesIn(NUM_THREADS)
    ),
    CArraySharedLong::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
    multithreaded_caching,
    CArraySharedDouble,
    ::testing::Combine(
        ::testing::ValuesIn(ARRAY_SIZES),
        ::testing::ValuesIn(NUM_THREADS)
    ),
    CArraySharedDouble::getTestCaseName
);
