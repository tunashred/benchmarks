#include "singlecore/matrix_as_array.hpp"
#include "utils/utils.hpp"
#include "utils/constants.hpp"

#include <cmath>

template <typename T>
void CMatrixArray<T>::SetUp() {
    size_t size = this->GetParam();

    ASSERT_NO_THROW(matrix_A = (T*) safe_malloc(size * size * sizeof(T)));
    ASSERT_NO_THROW(matrix_B = (T*) safe_malloc(size * size * sizeof(T)));
    ASSERT_NO_THROW(matrix_C = (T*) safe_malloc(size * size * sizeof(T)));

    memset(matrix_A, 3, size);
    memset(matrix_B, 3, size);
    memset(matrix_C, 0, size);
}

template <typename T>
void CMatrixArray<T>::TearDown() {
    free(matrix_A);
    free(matrix_B);
    free(matrix_C);

    threads.clear();
}

template <typename T>
void CMatrixArray<T>::naive_mul() {
    size_t matrixSize = GetParam();
    
    for (size_t i = 0; i < matrixSize; i++) {
        for (size_t j = 0; j < matrixSize; j++) {
            for (size_t k = 0; k < matrixSize; k++) {
                matrix_C[i * matrixSize + j] += matrix_A[i * matrixSize + k] * matrix_B[k * matrixSize + j];
            }
        }
    }
}

template <typename T>
void CMatrixArray<T>::optimized_mul() {
    size_t matrixSize = GetParam();
    
    for (size_t i = 0; i < matrixSize; i++) {
        for (size_t k = 0; k < matrixSize; k++) {
            T a_ik = matrix_A[i * matrixSize + k];
            for (size_t j = 0; j < matrixSize; j++) {
                matrix_C[i * matrixSize + j] += a_ik * matrix_B[k * matrixSize + j];
            }
        }
    }
}

using CMatrixArrayInt = CMatrixArray<int>;
using CMatrixArrayLong = CMatrixArray<long>;
using CMatrixArrayDouble = CMatrixArray<double>;

TEST_P(CMatrixArrayInt, DISABLED_NaiveMul) {
    naive_mul();
}

TEST_P(CMatrixArrayLong, DISABLED_NaiveMul) {
    naive_mul();
}

TEST_P(CMatrixArrayDouble, DISABLED_NaiveMul) {
    naive_mul();
}

TEST_P(CMatrixArrayInt, OptimizedMul) {
    optimized_mul();
}

TEST_P(CMatrixArrayLong, OptimizedMul) {
    optimized_mul();
}

TEST_P(CMatrixArrayDouble, OptimizedMul) {
    optimized_mul();
}

INSTANTIATE_TEST_SUITE_P(
    scalar_singlecore_caching,
    CMatrixArrayInt,
    ::testing::ValuesIn(MATRIX_SIZES_POW2),
    CMatrixArrayInt::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
    scalar_singlecore_caching,
    CMatrixArrayLong,
    ::testing::ValuesIn(MATRIX_SIZES_POW2),
    CMatrixArrayInt::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
    scalar_singlecore_caching,
    CMatrixArrayDouble,
    ::testing::ValuesIn(MATRIX_SIZES_POW2),
    CMatrixArrayInt::getTestCaseName
);
