#include "simd/multithreaded_iterate.hpp"
#include "utils/utils.hpp"

#include <immintrin.h>

template <typename T>
void AlignedArrayShared<T>::SetUp() {
    size_t size, numThreads;
    std::tie(size, numThreads) = this->GetParam();
    size_t totalSize = size * sizeof(T);
    if (totalSize % ALIGNMENT_32 != 0) {
        totalSize += ALIGNMENT_32 - (totalSize % ALIGNMENT_32);
    }
    array = static_cast<T*>(std::aligned_alloc(ALIGNMENT_32, totalSize));

    ASSERT_NE(array, nullptr) << "Unable to alloc array of size " << size;

    memset(array, 0, totalSize);
}

template <typename T>
void AlignedArrayShared<T>::TearDown() {
    free(array);
    threads.clear();
}

template <>
void sequential_iterate<int>(size_t start, size_t end, const AlignedArrayShared<int>* test) {
    for (size_t i = 0; i < LOOP_COUNT_200K; i++) {
        for (size_t j = start; j + SIMD_INT_WIDTH <= end; j += SIMD_INT_WIDTH) {
            __m256i vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(test->array + j));
    
            __m256i result = _mm256_add_epi32(vec, _mm256_set1_epi32(1));
    
            _mm256_store_si256(reinterpret_cast<__m256i*>(test->array + j), result);
        }
    }
}

template <>
void sequential_iterate<long>(size_t start, size_t end, const AlignedArrayShared<long>* test) {
    for (size_t i = 0; i < LOOP_COUNT_200K; i++) {
        for (size_t j = start; j + SIMD_LONG_WIDTH <= end; j += SIMD_LONG_WIDTH) {
            __m256i vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(test->array + j));

            __m256i result = _mm256_add_epi64(vec, _mm256_set1_epi64x(1));

            _mm256_store_si256(reinterpret_cast<__m256i*>(test->array + j), result);
        }
    }
}

template <>
void sequential_iterate<double>(size_t start, size_t end, const AlignedArrayShared<double>* test) {
    for (size_t i = 0; i < LOOP_COUNT_200K; i++) {
        for (size_t j = start; j + SIMD_DOUBLE_WIDTH <= end; j += SIMD_DOUBLE_WIDTH) {
            __m256d vec = _mm256_load_pd(test->array + j);

            __m256d result = _mm256_add_pd(vec, _mm256_set1_pd(1));

            _mm256_store_pd(test->array + j, result);
        }
    }
}

template <typename T>
void AlignedArrayShared<T>::runTest(iterate_function<T> iterate) {
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

using AlignedArraySharedInt = AlignedArrayShared<int>;
using AlignedArraySharedLong = AlignedArrayShared<long>;
using AlignedArraySharedDouble = AlignedArrayShared<double>;

TEST_P(AlignedArraySharedInt, SequentialIterate) {
    this->runTest(sequential_iterate<int>);
}

TEST_P(AlignedArraySharedLong, SequentialIterate) {
    this->runTest(sequential_iterate<long>);
}

TEST_P(AlignedArraySharedDouble, SequentialIterate) {
    this->runTest(sequential_iterate<double>);
}

std::vector<size_t> simd_sizes = {192, 960, 9984, 99840, 1000128, 2000064};

std::vector<size_t> simd_threads = {2, 4, 8, 12};

INSTANTIATE_TEST_SUITE_P(
    multithreaded_caching,
    AlignedArraySharedInt,
    ::testing::Combine(
        ::testing::ValuesIn(simd_sizes),
        ::testing::ValuesIn(simd_threads)
    ),
    AlignedArraySharedInt::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
    multithreaded_caching,
    AlignedArraySharedLong,
    ::testing::Combine(
        ::testing::ValuesIn(simd_sizes),
        ::testing::ValuesIn(simd_threads)
    ),
    AlignedArraySharedLong::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
    multithreaded_caching,
    AlignedArraySharedDouble,
    ::testing::Combine(
        ::testing::ValuesIn(simd_sizes),
        ::testing::ValuesIn(simd_threads)
    ),
    AlignedArraySharedDouble::getTestCaseName
);
