#include "singlecore/matrix.hpp"
#include "utils/utils.hpp"

template <typename T>
void CMatrix<T>::SetUp() {
    size_t size = this->GetParam();

    ASSERT_NO_THROW(create_matrix(matrix_A, size));
    ASSERT_NO_THROW(create_matrix(matrix_B, size));
    ASSERT_NO_THROW(create_matrix(matrix_C, size));
}

template <typename T>
void CMatrix<T>::TearDown() {
    size_t size = this->GetParam();

    free_matrix(reinterpret_cast<void**&>(matrix_A), size);
    free_matrix(reinterpret_cast<void**&>(matrix_B), size);
    free_matrix(reinterpret_cast<void**&>(matrix_C), size);
}

template <typename T>
void CMatrix<T>::naive_mul() {
    size_t size = this->GetParam();

    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            for (size_t k = 0; k < size; k++) {
                matrix_C[i][j] += matrix_A[i][k] * matrix_B[k][j];
            }
        }
    }
}

template <typename T>
void CMatrix<T>::optimized_mul() {
    size_t size = this->GetParam();

    for (size_t i = 0; i < size; i++) {
        for (size_t k = 0; k < size; k++) {
            for (size_t j = 0; j < size; j++) {
                matrix_C[i][j] += matrix_A[i][k] * matrix_B[k][j];
            }
        }
    }
}

using CMatrixInt = CMatrix<int>;
using CMatrixLong = CMatrix<long>;
using CMatrixDouble = CMatrix<double>;

TEST_P(CMatrixInt, NaiveMul) {
    naive_mul();
}

TEST_P(CMatrixLong, NaiveMul) {
    naive_mul();
}

TEST_P(CMatrixDouble, NaiveMul) {
    naive_mul();
}

TEST_P(CMatrixInt, OptimizedMul) {
    optimized_mul();
}

TEST_P(CMatrixLong, OptimizedMul) {
    optimized_mul();
}

TEST_P(CMatrixDouble, OptimizedMul) {
    optimized_mul();
}

INSTANTIATE_TEST_SUITE_P(
    singlecore_caching,
    CMatrixInt,
    ::testing::Values(512, 1024, 2048, 4096, 8192),
    CMatrixInt::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
    singlecore_caching,
    CMatrixLong,
    ::testing::Values(512, 1024, 2048, 4096, 8192),
    CMatrixLong::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
    singlecore_caching,
    CMatrixDouble,
    ::testing::Values(512, 1024, 2048, 4096, 8192),
    CMatrixDouble::getTestCaseName
);
