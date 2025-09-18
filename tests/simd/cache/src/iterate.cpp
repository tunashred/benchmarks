#include "simd/iterate.hpp"
#include "utils/constants.hpp"

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

    memset(array, 0, totalSize);
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
    size_t numElems = SIMD_INT_WIDTH * 4;
    const __m256i increment = _mm256_set1_epi32(1);

    for (size_t i = 0; i < LOOP_COUNT_200K; i++) {
        for (size_t j = 0; j + numElems <= size; j += numElems) {
            __m256i vec0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(array + j));
            __m256i vec1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(array + j + SIMD_INT_WIDTH));
            __m256i vec2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(array + j + SIMD_INT_WIDTH * 2));
            __m256i vec3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(array + j + SIMD_INT_WIDTH * 3));
            
            vec0 = _mm256_add_epi32(vec0, increment);
            vec1 = _mm256_add_epi32(vec1, increment);
            vec2 = _mm256_add_epi32(vec2, increment);
            vec3 = _mm256_add_epi32(vec3, increment);
            
            _mm256_store_si256(reinterpret_cast<__m256i*>(array + j), vec0);
            _mm256_store_si256(reinterpret_cast<__m256i*>(array + j + SIMD_INT_WIDTH), vec1);
            _mm256_store_si256(reinterpret_cast<__m256i*>(array + j + SIMD_INT_WIDTH * 2), vec2);
            _mm256_store_si256(reinterpret_cast<__m256i*>(array + j + SIMD_INT_WIDTH * 3), vec3);
        }
    }
}

TEST_P(AlignedArrayLong, SequentialIterate) {
    size_t size = this->GetParam();

    for (size_t i = 0; i < LOOP_COUNT_200K; i++) {
        for (size_t j = 0; j + SIMD_LONG_WIDTH <= size; j += SIMD_LONG_WIDTH) {
            __m256i vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(array + j));

            __m256i result = _mm256_add_epi64(vec, _mm256_set1_epi64x(1));

            _mm256_store_si256(reinterpret_cast<__m256i*>(array + j), result);
        }
    }
}

TEST_P(AlignedArrayDouble, SequentialIterate) {
    size_t size = this->GetParam();

    for (size_t i = 0; i < LOOP_COUNT_200K; i++) {
        for (size_t j = 0; j + SIMD_DOUBLE_WIDTH <= size; j += SIMD_DOUBLE_WIDTH) {
            __m256d vec = _mm256_load_pd(array + j);

            __m256d result = _mm256_add_pd(vec, _mm256_set1_pd(1));

            _mm256_store_pd(array + j, result);
        }
    }
}

// TODO: move values to a vector
INSTANTIATE_TEST_SUITE_P(
    simd_singlecore_caching,
    AlignedArrayInt,
    ::testing::Values(12, 100, 1000, 10000, 100000, 1000000, 2000000, 3000000),
    AlignedArrayInt::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
    simd_singlecore_caching,
    AlignedArrayLong,
    ::testing::Values(12, 100, 1000, 10000, 100000, 1000000, 2000000, 3000000),
    AlignedArrayLong::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
    simd_singlecore_caching,
    AlignedArrayDouble,
    ::testing::Values(12, 100, 1000, 10000, 100000, 1000000, 2000000, 3000000),
    AlignedArrayDouble::getTestCaseName
);
