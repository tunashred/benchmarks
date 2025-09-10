#include "simd/multithreaded_iterate.hpp"
#include "utils/constants.hpp"

#include <immintrin.h>

template <typename T>
void AlignedArrayShared<T>::SetUp() {
    size_t size, numThreads;
    std::tie(size, numThreads) = this->GetParam();
    size_t totalSize = size * sizeof(T);
    if (totalSize % ALIGNMENT_32 != 0) {
        totalSize += ALIGNMENT_32 - (totalSize % ALIGNMENT_32);
    }
    array = static_cast<T*>(std::aligned_alloc(ALIGNMENT_32, totalSize));

    ASSERT_NE(array, nullptr) << "Unable to alloc array of size " << size;

    memset(array, 0, totalSize);
}

template <typename T>
void AlignedArrayShared<T>::TearDown() {
    free(array);
    threads.clear();
}

// TODO: add scalar loops to increment the remainder of arrays
template <>
void sequential_iterate<int>(size_t start, size_t end, const AlignedArrayShared<int>* test) {
    const __m256i one_vec = _mm256_set1_epi32(1);
    const size_t unroll_end = start + ((end - start) / (SIMD_INT_WIDTH * 4)) * (SIMD_INT_WIDTH * 4);
    
    for (size_t i = 0; i < LOOP_COUNT_200K; i++) {
        for (size_t j = start; j < unroll_end; j += SIMD_INT_WIDTH * 4) {
            __m256i vec1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(test->array + j));
            __m256i vec2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(test->array + j + SIMD_INT_WIDTH));
            __m256i vec3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(test->array + j + SIMD_INT_WIDTH * 2));
            __m256i vec4 = _mm256_load_si256(reinterpret_cast<const __m256i*>(test->array + j + SIMD_INT_WIDTH * 3));
            
            vec1 = _mm256_add_epi32(vec1, one_vec);
            vec2 = _mm256_add_epi32(vec2, one_vec);
            vec3 = _mm256_add_epi32(vec3, one_vec);
            vec4 = _mm256_add_epi32(vec4, one_vec);
            
            _mm256_store_si256(reinterpret_cast<__m256i*>(test->array + j), vec1);
            _mm256_store_si256(reinterpret_cast<__m256i*>(test->array + j + SIMD_INT_WIDTH), vec2);
            _mm256_store_si256(reinterpret_cast<__m256i*>(test->array + j + SIMD_INT_WIDTH * 2), vec3);
            _mm256_store_si256(reinterpret_cast<__m256i*>(test->array + j + SIMD_INT_WIDTH * 3), vec4);
        }
    }
}

template <>
void sequential_iterate<long>(size_t start, size_t end, const AlignedArrayShared<long>* test) {
    const __m256i one_vec = _mm256_set1_epi64x(1);
    const size_t unroll_end = start + ((end - start) / (SIMD_LONG_WIDTH * 4)) * (SIMD_LONG_WIDTH * 4);
    
    for (size_t i = 0; i < LOOP_COUNT_200K; i++) {
        for (size_t j = start; j < unroll_end; j += SIMD_LONG_WIDTH * 4) {
            __m256i vec1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(test->array + j));
            __m256i vec2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(test->array + j + SIMD_LONG_WIDTH));
            __m256i vec3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(test->array + j + SIMD_LONG_WIDTH * 2));
            __m256i vec4 = _mm256_load_si256(reinterpret_cast<const __m256i*>(test->array + j + SIMD_LONG_WIDTH * 3));
            
            vec1 = _mm256_add_epi64(vec1, one_vec);
            vec2 = _mm256_add_epi64(vec2, one_vec);
            vec3 = _mm256_add_epi64(vec3, one_vec);
            vec4 = _mm256_add_epi64(vec4, one_vec);
            
            _mm256_store_si256(reinterpret_cast<__m256i*>(test->array + j), vec1);
            _mm256_store_si256(reinterpret_cast<__m256i*>(test->array + j + SIMD_LONG_WIDTH), vec2);
            _mm256_store_si256(reinterpret_cast<__m256i*>(test->array + j + SIMD_LONG_WIDTH * 2), vec3);
            _mm256_store_si256(reinterpret_cast<__m256i*>(test->array + j + SIMD_LONG_WIDTH * 3), vec4);
        }
    }
}

template <>
void sequential_iterate<double>(size_t start, size_t end, const AlignedArrayShared<double>* test) {
    const __m256d one_vec = _mm256_set1_pd(1.0);
    const size_t unroll_end = start + ((end - start) / (SIMD_DOUBLE_WIDTH * 4)) * (SIMD_DOUBLE_WIDTH * 4);
    
    for (size_t i = 0; i < LOOP_COUNT_200K; i++) {
        for (size_t j = start; j < unroll_end; j += SIMD_DOUBLE_WIDTH * 4) {
            __m256d vec1 = _mm256_load_pd(test->array + j);
            __m256d vec2 = _mm256_load_pd(test->array + j + SIMD_DOUBLE_WIDTH);
            __m256d vec3 = _mm256_load_pd(test->array + j + SIMD_DOUBLE_WIDTH * 2);
            __m256d vec4 = _mm256_load_pd(test->array + j + SIMD_DOUBLE_WIDTH * 3);
            
            vec1 = _mm256_add_pd(vec1, one_vec);
            vec2 = _mm256_add_pd(vec2, one_vec);
            vec3 = _mm256_add_pd(vec3, one_vec);
            vec4 = _mm256_add_pd(vec4, one_vec);
            
            _mm256_store_pd(test->array + j, vec1);
            _mm256_store_pd(test->array + j + SIMD_DOUBLE_WIDTH, vec2);
            _mm256_store_pd(test->array + j + SIMD_DOUBLE_WIDTH * 2, vec3);
            _mm256_store_pd(test->array + j + SIMD_DOUBLE_WIDTH * 3, vec4);
        }
    }
}

template <typename T>
void AlignedArrayShared<T>::runTest(iterate_function<T> iterate) {
    size_t totalSize, numThreads;
    std::tie(totalSize, numThreads) = this->GetParam();

    size_t sliceSize = totalSize / numThreads, 
           remainder = totalSize % numThreads,
           start     = 0;
    
    for (size_t i = 0; i < numThreads; i++) {
        size_t currentSliceSize = sliceSize + (i < remainder ? 1 : 0);
        size_t end = start + currentSliceSize;
        
        threads.emplace_back(iterate, start, end, this);
        start = end;
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

using AlignedArraySharedInt = AlignedArrayShared<int>;
using AlignedArraySharedLong = AlignedArrayShared<long>;
using AlignedArraySharedDouble = AlignedArrayShared<double>;

TEST_P(AlignedArraySharedInt, SequentialIterate) {
    this->runTest(sequential_iterate<int>);
}

TEST_P(AlignedArraySharedLong, SequentialIterate) {
    this->runTest(sequential_iterate<long>);
}

TEST_P(AlignedArraySharedDouble, SequentialIterate) {
    this->runTest(sequential_iterate<double>);
}

INSTANTIATE_TEST_SUITE_P(
    simd_multithreaded_caching,
    AlignedArraySharedInt,
    ::testing::Combine(
        ::testing::ValuesIn(ARRAY_SIZES),
        ::testing::ValuesIn(NUM_THREADS)
    ),
    AlignedArraySharedInt::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
    simd_multithreaded_caching,
    AlignedArraySharedLong,
    ::testing::Combine(
        ::testing::ValuesIn(ARRAY_SIZES),
        ::testing::ValuesIn(NUM_THREADS)
    ),
    AlignedArraySharedLong::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
    simd_multithreaded_caching,
    AlignedArraySharedDouble,
    ::testing::Combine(
        ::testing::ValuesIn(ARRAY_SIZES),
        ::testing::ValuesIn(NUM_THREADS)
    ),
    AlignedArraySharedDouble::getTestCaseName
);
