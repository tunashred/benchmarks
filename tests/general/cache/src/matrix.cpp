#include "matrix.hpp"
#include "utils/utils.hpp"

template <typename T>
void create_matrix(T**& matrix, size_t size) {
    size_t rowSize = size * sizeof(T);

    try {
        matrix = (T**) safe_malloc(size * sizeof(T*));
    
        for (size_t i = 0; i < size; i++) {
            matrix[i] = (T*) safe_malloc(rowSize);
            memset(matrix[i], 3, rowSize);
        }
    } catch (const std::bad_alloc& e) {
        std::cout << "Allocation failed: " << e.what() << '\n';
    }
}

template <typename T>
void CMatrix<T>::SetUp() {
    size_t size = this->GetParam();

    create_matrix(matrix_A, size);
    create_matrix(matrix_B, size);
    create_matrix(matrix_C, size);
}

void free_matrix(void**& matrix, size_t size) {
    for (size_t i = 0; i < size; i++) {
        free(matrix[i]);
    }
    free(matrix);
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

template <typename T>
void CMatrix<T>::block_mul() {
    size_t size = this->GetParam();

    for (size_t ii = 0; ii < size; ii += BLOCK_SIZE) {
        for (size_t jj = 0; jj < size; jj += BLOCK_SIZE) {
            for (size_t kk = 0; kk < size; kk += BLOCK_SIZE) {
                // Multiply the blocks
                for (size_t i = ii; i < ii + BLOCK_SIZE && i < size; i++) {
                    for (size_t k = kk; k < kk + BLOCK_SIZE && k < size; k++) {
                        for (size_t j = jj; j < jj + BLOCK_SIZE && j < size; j++) {
                            matrix_C[i][j] += matrix_A[i][k] * matrix_B[k][j];
                        }
                    }
                }
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

TEST_P(CMatrixInt, BlockMul) {
    block_mul();
}

TEST_P(CMatrixLong, BlockMul) {
    block_mul();
}

TEST_P(CMatrixDouble, BlockMul) {
    block_mul();
}

INSTANTIATE_TEST_SUITE_P(
    GeneralCaching,
    CMatrixInt,
    ::testing::Values(512, 1024, 2048, 4096, 8192),
    CMatrixInt::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
    GeneralCaching,
    CMatrixLong,
    ::testing::Values(512, 1024, 2048, 4096, 8192),
    CMatrixLong::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
    GeneralCaching,
    CMatrixDouble,
    ::testing::Values(512, 1024, 2048, 4096, 8192),
    CMatrixDouble::getTestCaseName
);
