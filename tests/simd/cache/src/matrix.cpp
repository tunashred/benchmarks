#include "simd/matrix.hpp"
#include "utils/constants.hpp"

#include <immintrin.h>

template <typename T>
void AlignedMatrix<T>::SetUp() {
    size_t size = this->GetParam();
    size_t totalSize = size * size * sizeof(T);
    if (totalSize % ALIGNMENT_32 != 0) {
        totalSize += ALIGNMENT_32 - (totalSize % ALIGNMENT_32);
    }
    matrix_A = static_cast<T*>(std::aligned_alloc(ALIGNMENT_32, totalSize));
    matrix_B = static_cast<T*>(std::aligned_alloc(ALIGNMENT_32, totalSize));
    matrix_C = static_cast<T*>(std::aligned_alloc(ALIGNMENT_32, totalSize));

    memset(matrix_A, 3, totalSize);
    memset(matrix_B, 3, totalSize);
    memset(matrix_C, 0, totalSize);
}

template <typename T>
void AlignedMatrix<T>::TearDown() {
    free(matrix_A);
    free(matrix_B);
    free(matrix_C);
}

template <>
void AlignedMatrix<int>::naive_mul() {
    size_t size = this->GetParam();
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j + SIMD_INT_WIDTH <= size; j += SIMD_INT_WIDTH) {
            __m256i sum = _mm256_setzero_si256();
            for (size_t k = 0; k < size; k++) {
                __m256i a = _mm256_set1_epi32(matrix_A[i * size + k]);

                __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(&matrix_B[k * size + j]));

                sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(a, b));
            }
            _mm256_store_si256(reinterpret_cast<__m256i*>(&matrix_C[i * size + j]), sum);
        }
    }
}

template <>
void AlignedMatrix<int>::optimized_mul() {
    size_t size = this->GetParam();
    for (size_t i = 0; i < size; i++) {
        for (size_t k = 0; k < size; k++) {
            __m256i a = _mm256_set1_epi32(matrix_A[i * size + k]);
            for (size_t j = 0; j + SIMD_INT_WIDTH <= size; j += SIMD_INT_WIDTH) { 
                __m256i c = _mm256_load_si256(reinterpret_cast<__m256i*>(&matrix_C[i * size + j]));

                __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(&matrix_B[k * size + j]));

                c = _mm256_add_epi32(c, _mm256_mullo_epi32(a, b));

                _mm256_store_si256(reinterpret_cast<__m256i*>(&matrix_C[i * size + j]), c);
            }
        }
    }
}

template <>
void AlignedMatrix<long>::optimized_mul() {
    size_t size = this->GetParam();
    for (size_t i = 0; i < size; i++) {
        for (size_t k = 0; k < size; k++) {
            __m256i a = _mm256_set1_epi64x(matrix_A[i * size + k]);
            for (size_t j = 0; j + SIMD_LONG_WIDTH <= size; j += SIMD_LONG_WIDTH) { 
                __m256i c = _mm256_load_si256(reinterpret_cast<__m256i*>(&matrix_C[i * size + j]));

                __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(&matrix_B[k * size + j]));

                // the multiply is on int because my CPU does not support AVX512
                c = _mm256_add_epi64(c, _mm256_mullo_epi32(a, b));

                _mm256_store_si256(reinterpret_cast<__m256i*>(&matrix_C[i * size + j]), c);
            }
        }
    }
}

template <>
void AlignedMatrix<double>::optimized_mul() {
    size_t size = this->GetParam();
    for (size_t i = 0; i < size; i++) {
        for (size_t k = 0; k < size; k++) {
            __m256d a = _mm256_set1_pd(matrix_A[i * size + k]);
            for (size_t j = 0; j + SIMD_DOUBLE_WIDTH <= size; j += SIMD_DOUBLE_WIDTH) { 
                __m256d c = _mm256_load_pd(&matrix_C[i * size + j]);

                __m256d b = _mm256_load_pd(&matrix_B[k * size + j]);

                c = _mm256_fmadd_pd(a, b, c);

                _mm256_store_pd(&matrix_C[i * size + j], c);
            }
        }
    }
}

using AlignedMatrixInt = AlignedMatrix<int>;
using AlignedMatrixLong = AlignedMatrix<long>;
using AlignedMatrixDouble = AlignedMatrix<double>;

TEST_P(AlignedMatrixInt, DISABLED_NaiveMul) {
    naive_mul();
}

TEST_P(AlignedMatrixInt, OptimizedMul) {
    optimized_mul();
}

TEST_P(AlignedMatrixLong, OptimizedMul) {
    optimized_mul();
}

TEST_P(AlignedMatrixDouble, OptimizedMul) {
    optimized_mul();
}

INSTANTIATE_TEST_SUITE_P(
    simd_singlecore_caching,
    AlignedMatrixInt,
    ::testing::ValuesIn(MATRIX_SIZES_POW2),
    AlignedMatrixInt::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
    simd_singlecore_caching,
    AlignedMatrixLong,
    ::testing::ValuesIn(MATRIX_SIZES_POW2),
    AlignedMatrixLong::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
    simd_singlecore_caching,
    AlignedMatrixDouble,
    ::testing::ValuesIn(MATRIX_SIZES_POW2),
    AlignedMatrixDouble::getTestCaseName
);
