#include "multithreaded/matrix_as_array.hpp"
#include "utils/utils.hpp"
#include "utils/constants.hpp"

#include <cmath>

template <typename T>
void CMatrixArrayShared<T>::SetUp() {
    size_t size, numThreads;
    std::tie(size, numThreads) = this->GetParam();

    ASSERT_NO_THROW(matrix_A = (T*) safe_malloc(size * size * sizeof(T)));
    ASSERT_NO_THROW(matrix_B = (T*) safe_malloc(size * size * sizeof(T)));
    ASSERT_NO_THROW(matrix_C = (T*) safe_malloc(size * size * sizeof(T)));

    memset(matrix_A, 3, size);
    memset(matrix_B, 3, size);
    memset(matrix_C, 0, size);
}

template <typename T>
void CMatrixArrayShared<T>::TearDown() {
    free(matrix_A);
    free(matrix_B);
    free(matrix_C);

    threads.clear();
}

template <typename T>
void naive_mul(size_t startRow, size_t endRow, size_t startCol, size_t endCol, const CMatrixArrayShared<T>* test) {
    size_t matrixSize, numThreads;
    std::tie(matrixSize, numThreads) = test->GetParam();
    
    for (size_t i = startRow; i < endRow; i++) {
        for (size_t j = startCol; j < endCol; j++) {
            for (size_t k = 0; k < matrixSize; k++) {
                test->matrix_C[i * matrixSize + j] += test->matrix_A[i * matrixSize + k] * test->matrix_B[k * matrixSize + j];
            }
        }
    }
}

template <typename T>
void optimized_mul(size_t startRow, size_t endRow, size_t startCol, size_t endCol, const CMatrixArrayShared<T>* test) {
    size_t matrixSize, numThreads;
    std::tie(matrixSize, numThreads) = test->GetParam();
    
    for (size_t i = startRow; i < endRow; i++) {
        for (size_t k = 0; k < matrixSize; k++) {
            T a_ik = test->matrix_A[i * matrixSize + k];
            for (size_t j = startCol; j < endCol; j++) {
                test->matrix_C[i * matrixSize + j] += a_ik * test->matrix_B[k * matrixSize + j];
            }
        }
    }
}

template <typename T>
void CMatrixArrayShared<T>::runTest(mul_function<T> mul) {
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

using CMatrixArraySharedInt = CMatrixArrayShared<int>;
using CMatrixArraySharedLong = CMatrixArrayShared<long>;
using CMatrixArraySharedDouble = CMatrixArrayShared<double>;

TEST_P(CMatrixArraySharedInt, DISABLED_NaiveMul) {
    this->runTest(::naive_mul<int>);
}

TEST_P(CMatrixArraySharedLong, DISABLED_NaiveMul) {
    this->runTest(::naive_mul<long>);
}

TEST_P(CMatrixArraySharedDouble, DISABLED_NaiveMul) {
    this->runTest(::naive_mul<double>);
}

TEST_P(CMatrixArraySharedInt, OptimizedMul) {
    this->runTest(::optimized_mul<int>);
}

TEST_P(CMatrixArraySharedLong, OptimizedMul) {
    this->runTest(::optimized_mul<long>);
}

TEST_P(CMatrixArraySharedDouble, OptimizedMul) {
    this->runTest(::optimized_mul<double>);
}

INSTANTIATE_TEST_SUITE_P(
    multithreaded_caching,
    CMatrixArraySharedInt,
    ::testing::Combine(
        ::testing::ValuesIn(MATRIX_SIZES_POW2),
        ::testing::ValuesIn(NUM_THREADS)
    ),
    CMatrixArraySharedInt::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
    multithreaded_caching,
    CMatrixArraySharedLong,
    ::testing::Combine(
        ::testing::ValuesIn(MATRIX_SIZES_POW2),
        ::testing::ValuesIn(NUM_THREADS)
    ),
    CMatrixArraySharedLong::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
    multithreaded_caching,
    CMatrixArraySharedDouble,
    ::testing::Combine(
        ::testing::ValuesIn(MATRIX_SIZES_POW2),
        ::testing::ValuesIn(NUM_THREADS)
    ),
    CMatrixArraySharedDouble::getTestCaseName
);
