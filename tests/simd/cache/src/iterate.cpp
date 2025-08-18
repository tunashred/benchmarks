#include "simd/iterate.hpp"
#include <immintrin.h>
#include <cstdlib>

template <typename T>
void AlignedArray<T>::SetUp() {
    size_t size = this->GetParam();
    size_t totalSize = size * sizeof(T);
    if (totalSize % ALIGNMENT_32 != 0) {
        totalSize += ALIGNMENT_32 - (totalSize % ALIGNMENT_32);
    }
    array = static_cast<T*>(std::aligned_alloc(ALIGNMENT_32, totalSize));

    ASSERT_NE(array, nullptr) << "Unable to alloc array of size " << size;

    for (size_t i = 0; i < size; i++) {
        array[i] = static_cast<T>(0);
    }
}

template <typename T>
void AlignedArray<T>::TearDown() {
    std::free(array);
}

using AlignedArrayInt = AlignedArray<int>;
using AlignedArrayLong = AlignedArray<long>;
using AlignedArrayDouble = AlignedArray<double>;

TEST_P(AlignedArrayInt, SequentialIterate) {
    size_t size = this->GetParam();

    for (size_t i = 0; i < LOOP_COUNT; i++) {
        for (size_t j = 0; j + SIMD_INT_WIDTH <= size; j += SIMD_INT_WIDTH) {
            __m256i vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(array + j));
    
            __m256i result = _mm256_add_epi32(vec, _mm256_set1_epi32(1));
    
            _mm256_store_si256(reinterpret_cast<__m256i*>(array + j), result);
        }
    }
}

TEST_P(AlignedArrayLong, SequentialIterate) {
    size_t size = this->GetParam();

    for (size_t i = 0; i < LOOP_COUNT; i++) {
        for (size_t j = 0; j + SIMD_LONG_WIDTH <= size; j += SIMD_LONG_WIDTH) {
            __m256i vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(array + j));

            __m256i result = _mm256_add_epi64(vec, _mm256_set1_epi64x(1));

            _mm256_store_si256(reinterpret_cast<__m256i*>(array + j), result);
        }
    }
}

TEST_P(AlignedArrayDouble, SequentialIterate) {
    size_t size = this->GetParam();

    for (size_t i = 0; i < LOOP_COUNT; i++) {
        for (size_t j = 0; j + SIMD_DOUBLE_WIDTH <= size; j += SIMD_DOUBLE_WIDTH) {
            __m256d vec = _mm256_load_pd(array + j);

            __m256d result = _mm256_add_pd(vec, _mm256_set1_pd(1));

            _mm256_store_pd(array + j, result);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    simd_singlecore_iteration,
    AlignedArrayInt,
    ::testing::Values(12, 100, 1000, 10000, 100000, 1000000, 2000000, 3000000),
    AlignedArrayInt::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
    simd_singlecore_iteration,
    AlignedArrayLong,
    ::testing::Values(12, 100, 1000, 10000, 100000, 1000000, 2000000, 3000000),
    AlignedArrayLong::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
    simd_singlecore_iteration,
    AlignedArrayDouble,
    ::testing::Values(12, 100, 1000, 10000, 100000, 1000000, 2000000, 3000000),
    AlignedArrayDouble::getTestCaseName
);
