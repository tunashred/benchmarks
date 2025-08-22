#include "simd/multithreaded_matrix.hpp"
#include "utils/constants.hpp"

#include <immintrin.h>

template <typename T>
void AlignedMatrixShared<T>::SetUp() {
    size_t size, numThreads;
    std::tie(size, numThreads) = this->GetParam();
    size_t totalSize = size * size * sizeof(T);
    if (totalSize % ALIGNMENT_32 != 0) {
        totalSize += ALIGNMENT_32 - (totalSize % ALIGNMENT_32);
    }
    matrix_A = static_cast<T*>(std::aligned_alloc(ALIGNMENT_32, totalSize));
    matrix_B = static_cast<T*>(std::aligned_alloc(ALIGNMENT_32, totalSize));
    matrix_C = static_cast<T*>(std::aligned_alloc(ALIGNMENT_32, totalSize));

    memset(matrix_A, 3, totalSize);
    memset(matrix_B, 3, totalSize);
    memset(matrix_C, 3, totalSize);
}

template <typename T>
void AlignedMatrixShared<T>::TearDown() {
    free(matrix_A);
    free(matrix_B);
    free(matrix_C);

    threads.clear();
}

template <>
void optimized_mul<int>(size_t startRow, size_t endRow, size_t startCol, size_t endCol, const AlignedMatrixShared<int>* test) {
    size_t matrixSize, numThreads;
    std::tie(matrixSize, numThreads) = test->GetParam();
    
    for (size_t i = startRow; i < endRow; i++) {
        for (size_t k = 0; k < matrixSize; k++) {
            __m256i a = _mm256_set1_epi32(test->matrix_A[i * matrixSize + k]);
            for (size_t j = startCol; j + SIMD_INT_WIDTH <= endCol; j += SIMD_INT_WIDTH) { 
                __m256i c = _mm256_load_si256(reinterpret_cast<__m256i*>(&test->matrix_C[i * matrixSize + j]));
                __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(&test->matrix_B[k * matrixSize + j]));
                c = _mm256_add_epi32(c, _mm256_mullo_epi32(a, b));
                _mm256_store_si256(reinterpret_cast<__m256i*>(&test->matrix_C[i * matrixSize + j]), c);
            }
        }
    }
}

template <>
void optimized_mul<long>(size_t startRow, size_t endRow, size_t startCol, size_t endCol, const AlignedMatrixShared<long>* test) {
    size_t matrixSize, numThreads;
    std::tie(matrixSize, numThreads) = test->GetParam();
    
    for (size_t i = startRow; i < endRow; i++) {
        for (size_t k = 0; k < matrixSize; k++) {
            __m256i a = _mm256_set1_epi64x(test->matrix_A[i * matrixSize + k]);
            for (size_t j = startCol; j + SIMD_LONG_WIDTH <= endCol; j += SIMD_LONG_WIDTH) { 
                __m256i c = _mm256_load_si256(reinterpret_cast<__m256i*>(&test->matrix_C[i * matrixSize + j]));
                __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(&test->matrix_B[k * matrixSize + j]));
        
                // the multiply is on int because my CPU does not support AVX512
                c = _mm256_add_epi64(c, _mm256_mullo_epi32(a, b));
                _mm256_store_si256(reinterpret_cast<__m256i*>(&test->matrix_C[i * matrixSize + j]), c);
            }
        }
    }
}

template <>
void optimized_mul<double>(size_t startRow, size_t endRow, size_t startCol, size_t endCol, const AlignedMatrixShared<double>* test) {
    size_t matrixSize, numThreads;
    std::tie(matrixSize, numThreads) = test->GetParam();
    
    for (size_t i = startRow; i < endRow; i++) {
        for (size_t k = 0; k < matrixSize; k++) {
            __m256d a = _mm256_set1_pd(test->matrix_A[i * matrixSize + k]);
            for (size_t j = startCol; j + SIMD_DOUBLE_WIDTH <= endCol; j += SIMD_DOUBLE_WIDTH) { 
                __m256d c = _mm256_load_pd(&test->matrix_C[i * matrixSize + j]);
                __m256d b = _mm256_load_pd(&test->matrix_B[k * matrixSize + j]);
                c = _mm256_add_pd(c, _mm256_mul_pd(a, b));
                _mm256_store_pd(&test->matrix_C[i * matrixSize + j], c);
            }
        }
    }
}

template <typename T>
void AlignedMatrixShared<T>::runTest(mul_function<T> mul) {
    size_t matrixSize, numThreads;
    std::tie(matrixSize, numThreads) = this->GetParam();
    
    size_t rowsPerThread = matrixSize / numThreads;
    size_t remainderRows = matrixSize % numThreads;
    
    for (size_t threadId = 0; threadId < numThreads; ++threadId) {
        size_t startRow = threadId * rowsPerThread;
        size_t endRow = (threadId + 1) * rowsPerThread;
        
        if (threadId < remainderRows) {
            startRow += threadId;
            endRow += threadId + 1;
        } else {
            startRow += remainderRows;
            endRow += remainderRows;
        }
        
        threads.emplace_back(mul, startRow, endRow, 0, matrixSize, this);
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
}

using AlignedMatrixSharedInt = AlignedMatrixShared<int>;
using AlignedMatrixSharedLong = AlignedMatrixShared<long>;
using AlignedMatrixSharedDouble = AlignedMatrixShared<double>;

TEST_P(AlignedMatrixSharedInt, OptimizedMul) {
    runTest(::optimized_mul<int>);
}

TEST_P(AlignedMatrixSharedLong, OptimizedMul) {
    runTest(::optimized_mul<long>);
}

TEST_P(AlignedMatrixSharedDouble, OptimizedMul) {
    runTest(::optimized_mul<double>);
}

std::vector<size_t> simd_matrix_sizes = {512, 1024, 2048, 4096, 8192};

std::vector<size_t> simd_matrix_threads = {2, 4, 8, 12};

INSTANTIATE_TEST_SUITE_P(
    simd_multithreaded_matrix,
    AlignedMatrixSharedInt,
    ::testing::Combine(
        ::testing::ValuesIn(simd_matrix_sizes),
        ::testing::ValuesIn(simd_matrix_threads)
    ),
    AlignedMatrixSharedInt::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
    simd_multithreaded_matrix,
    AlignedMatrixSharedLong,
    ::testing::Combine(
        ::testing::ValuesIn(simd_matrix_sizes),
        ::testing::ValuesIn(simd_matrix_threads)
    ),
    AlignedMatrixSharedLong::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
    simd_multithreaded_matrix,
    AlignedMatrixSharedDouble,
    ::testing::Combine(
        ::testing::ValuesIn(simd_matrix_sizes),
        ::testing::ValuesIn(simd_matrix_threads)
    ),
    AlignedMatrixSharedDouble::getTestCaseName
);
