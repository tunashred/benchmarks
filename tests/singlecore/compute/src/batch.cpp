#include "singlecore/batch.hpp"
#include "utils/utils.hpp"
#include "utils/constants.hpp"

template <typename T>
void CArrayComputeBatch<T>::SetUp() {
    size_t size = this->GetParam();
    size_t totalSize = size * sizeof *array;

    array = (T*) safe_malloc(totalSize);
    ASSERT_NE(array, nullptr) << "Unable to alloc array of size " << size;

    // warming up the memory
    memset(array, 1, totalSize);
}

template <typename T>
void CArrayComputeBatch<T>::TearDown() {
    free(array);
}

template <typename T>
void CArrayComputeBatch<T>::batch_add() {
    size_t size = GetParam();
    for (size_t i = 0; i < LOOP_COUNT_400K; i++) {
        for (size_t j = 0; j < size; j++) {
            array[j] += array[j];
        }
    }
}

template <typename T>
void CArrayComputeBatch<T>::batch_mul() {
    size_t size = GetParam();
    for (size_t i = 0; i < LOOP_COUNT_200K; i++) {
        for (size_t j = 0; j < LOOP_COUNT_18; j++) {
            for (size_t k = 0; k < size; k++) {
                array[k] *= 3 ;
            }
        }
    }
}

using CArrayComputeBatchInt = CArrayComputeBatch<int>;
using CArrayComputeBatchLong = CArrayComputeBatch<long>;
using CArrayComputeBatchDouble = CArrayComputeBatch<double>;

TEST_P(CArrayComputeBatchInt, BatchAdd) {
    batch_add();
}

TEST_P(CArrayComputeBatchLong, BatchAdd) {
    batch_add();
}

TEST_P(CArrayComputeBatchDouble, BatchAdd) {
    batch_add();
}

TEST_P(CArrayComputeBatchInt, BatchMul) {
    batch_mul();
}

TEST_P(CArrayComputeBatchLong, BatchMul) {
    batch_mul();
}

TEST_P(CArrayComputeBatchDouble, BatchMul) {
    batch_mul();
}

INSTANTIATE_TEST_SUITE_P(
    singlecore_compute,
    CArrayComputeBatchInt,
    ::testing::ValuesIn(small_pow2),
    CArrayComputeBatchInt::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
    singlecore_compute,
    CArrayComputeBatchLong,
    ::testing::ValuesIn(small_pow2),
    CArrayComputeBatchLong::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
    singlecore_compute,
    CArrayComputeBatchDouble,
    ::testing::ValuesIn(small_pow2),
    CArrayComputeBatchDouble::getTestCaseName
);
