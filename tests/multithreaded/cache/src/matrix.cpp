#include "multithreaded/matrix.hpp"
#include "utils/utils.hpp"
#include "utils/constants.hpp"

#include <cmath>

template <typename T>
void CMatrixShared<T>::SetUp() {
    size_t size, numThreads;
    std::tie(size, numThreads) = this->GetParam();

    ASSERT_NO_THROW(create_matrix(matrix_A, size));
    ASSERT_NO_THROW(create_matrix(matrix_B, size));
    ASSERT_NO_THROW(create_matrix(matrix_C, size));
}

template <typename T>
void CMatrixShared<T>::TearDown() {
    size_t size, numThreads;
    std::tie(size, numThreads) = this->GetParam();

    free_matrix(reinterpret_cast<void**&>(matrix_A), size);
    free_matrix(reinterpret_cast<void**&>(matrix_B), size);
    free_matrix(reinterpret_cast<void**&>(matrix_C), size);
}

template <typename T>
void naive_mul(size_t startRow, size_t endRow, size_t startCol, size_t endCol, const CMatrixShared<T>* test) {
    size_t matrixSize, numThreads;
    std::tie(matrixSize, numThreads) = test->GetParam();
    
    for (size_t i = startRow; i < endRow; i++) {
        for (size_t j = startCol; j < endCol; j++) {
            for (size_t k = 0; k < matrixSize; k++) {
                test->matrix_C[i][j] += test->matrix_A[i][k] * test->matrix_B[k][j];
            }
        }
    }
}

template <typename T>
void optimized_mul(size_t startRow, size_t endRow, size_t startCol, size_t endCol, const CMatrixShared<T>* test) {
    size_t matrixSize, numThreads;
    std::tie(matrixSize, numThreads) = test->GetParam();
    
    for (size_t i = startRow; i < endRow; i++) {
        for (size_t k = 0; k < matrixSize; k++) {
            for (size_t j = startCol; j < endCol; j++) {
                test->matrix_C[i][j] += test->matrix_A[i][k] * test->matrix_B[k][j];
            }
        }
    }
}

template <typename T>
void CMatrixShared<T>::runTest(mul_function<T> mul) {
    size_t matrixSize, numThreads;
    std::tie(matrixSize, numThreads) = this->GetParam();

    size_t gridSize = static_cast<size_t>(std::sqrt(numThreads));
    if (gridSize * gridSize != numThreads) {
        size_t rowsPerThread = matrixSize / numThreads;
        size_t remainder = matrixSize % numThreads;
        size_t startRow = 0;
        
        for (size_t i = 0; i < numThreads; i++) {
            size_t currentRows = rowsPerThread + (i < remainder ? 1 : 0);
            size_t endRow = startRow + currentRows;
            
            threads.emplace_back(mul, startRow, endRow, 0, matrixSize, this);
            startRow = endRow;
        }
    } else {
        size_t blockRows = matrixSize / gridSize;
        size_t blockCols = matrixSize / gridSize;
        size_t rowRemainder = matrixSize % gridSize;
        size_t colRemainder = matrixSize % gridSize;
        
        for (size_t i = 0; i < gridSize; i++) {
            for (size_t j = 0; j < gridSize; j++) {
                size_t startRow = i * blockRows + std::min(i, rowRemainder);
                size_t endRow = startRow + blockRows + (i < rowRemainder ? 1 : 0);
                
                size_t startCol = j * blockCols + std::min(j, colRemainder);
                size_t endCol = startCol + blockCols + (j < colRemainder ? 1 : 0);
                
                threads.emplace_back(mul, startRow, endRow, startCol, endCol, this);
            }
        }
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

using CMatrixSharedInt = CMatrixShared<int>;
using CMatrixSharedLong = CMatrixShared<long>;
using CMatrixSharedDouble = CMatrixShared<double>;

TEST_P(CMatrixSharedInt, DISABLED_NaiveMul) {
    this->runTest(::naive_mul<int>);
}

TEST_P(CMatrixSharedLong, DISABLED_NaiveMul) {
    this->runTest(::naive_mul<long>);
}

TEST_P(CMatrixSharedDouble, DISABLED_NaiveMul) {
    this->runTest(::naive_mul<double>);
}

TEST_P(CMatrixSharedInt, OptimizedMul) {
    this->runTest(::optimized_mul<int>);
}

TEST_P(CMatrixSharedLong, OptimizedMul) {
    this->runTest(::optimized_mul<long>);
}

TEST_P(CMatrixSharedDouble, OptimizedMul) {
    this->runTest(::optimized_mul<double>);
}

INSTANTIATE_TEST_SUITE_P(
    multithreaded_caching,
    CMatrixSharedInt,
    ::testing::Combine(
        ::testing::ValuesIn(MATRIX_SIZES_POW2),
        ::testing::ValuesIn(NUM_THREADS)
    ),
    CMatrixSharedInt::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
    multithreaded_caching,
    CMatrixSharedLong,
    ::testing::Combine(
        ::testing::ValuesIn(MATRIX_SIZES_POW2),
        ::testing::ValuesIn(NUM_THREADS)
    ),
    CMatrixSharedLong::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
    multithreaded_caching,
    CMatrixSharedDouble,
    ::testing::Combine(
        ::testing::ValuesIn(MATRIX_SIZES_POW2),
        ::testing::ValuesIn(NUM_THREADS)
    ),
    CMatrixSharedDouble::getTestCaseName
);
