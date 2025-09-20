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
    size_t numElems = SIMD_LONG_WIDTH * 4;
    const __m256i increment = _mm256_set1_epi64x(1);

    for (size_t i = 0; i < LOOP_COUNT_200K; i++) {
        for (size_t j = 0; j + numElems <= size; j += numElems) {
            __m256i vec0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(array + j));
            __m256i vec1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(array + j + SIMD_LONG_WIDTH));
            __m256i vec2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(array + j + SIMD_LONG_WIDTH * 2));
            __m256i vec3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(array + j + SIMD_LONG_WIDTH * 3));

            vec0 = _mm256_add_epi64(vec0, increment);
            vec1 = _mm256_add_epi64(vec1, increment);
            vec2 = _mm256_add_epi64(vec2, increment);
            vec3 = _mm256_add_epi64(vec3, increment);

            _mm256_store_si256(reinterpret_cast<__m256i*>(array + j), vec0);
            _mm256_store_si256(reinterpret_cast<__m256i*>(array + j + SIMD_LONG_WIDTH), vec1);
            _mm256_store_si256(reinterpret_cast<__m256i*>(array + j + SIMD_LONG_WIDTH * 2), vec2);
            _mm256_store_si256(reinterpret_cast<__m256i*>(array + j + SIMD_LONG_WIDTH * 3), vec3);
        }
    }
}

TEST_P(AlignedArrayDouble, SequentialIterate) {
    size_t size = this->GetParam();
    size_t numElems = SIMD_DOUBLE_WIDTH * 4;
    const __m256d increment = _mm256_set1_pd(1);

    for (size_t i = 0; i < LOOP_COUNT_200K; i++) {
        for (size_t j = 0; j + numElems <= size; j += numElems) {
            __m256d vec0 = _mm256_load_pd(array + j);
            __m256d vec1 = _mm256_load_pd(array + j + SIMD_DOUBLE_WIDTH);
            __m256d vec2 = _mm256_load_pd(array + j + SIMD_DOUBLE_WIDTH * 2);
            __m256d vec3 = _mm256_load_pd(array + j + SIMD_DOUBLE_WIDTH * 3);

            vec0 = _mm256_add_pd(vec0, increment);
            vec1 = _mm256_add_pd(vec1, increment);
            vec2 = _mm256_add_pd(vec2, increment);
            vec3 = _mm256_add_pd(vec3, increment);

            _mm256_store_pd(array + j, vec0);
            _mm256_store_pd(array + j + SIMD_DOUBLE_WIDTH, vec1);
            _mm256_store_pd(array + j + SIMD_DOUBLE_WIDTH * 2, vec2);
            _mm256_store_pd(array + j + SIMD_DOUBLE_WIDTH * 3, vec3);
        }
    }
}

// TODO: move values to a vector
// TOOD 2: add 4'000'000
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
