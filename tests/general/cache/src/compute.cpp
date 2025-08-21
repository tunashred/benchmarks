#include "singlecore/compute.hpp"
#include "utils/utils.hpp"

template <typename T>
void CArrayCompute<T>::SetUp() {
    size_t size = this->GetParam();
    size_t totalSize = size * sizeof *array;

    array = (T*) safe_malloc(totalSize);
    ASSERT_NE(array, nullptr) << "Unable to alloc array of size " << size;

    // warming up the memory
    memset(array, 1, totalSize);
}

template <typename T>
void CArrayCompute<T>::TearDown() {
    free(array);
}

template <typename T>
void CArrayCompute<T>::batch_add() {
    size_t size = GetParam();
    for (size_t i = 0; i < LOOP_COUNT_400K; i++) {
        for (size_t j = 0; j < size; j++) {
            array[j] += array[j];
        }
    }
}

template <typename T>
void CArrayCompute<T>::batch_mul() {
    size_t size = GetParam();
    for (size_t i = 0; i < LOOP_COUNT_200K; i++) {
        for (size_t j = 0; j < LOOP_COUNT_18; j++) {
            for (size_t k = 0; k < size; k++) {
                array[k] *= 3 ;
            }
        }
    }
}

using CArrayComputeInt = CArrayCompute<int>;
using CArrayComputeLong = CArrayCompute<long>;
using CArrayComputeDouble = CArrayCompute<double>;

TEST_P(CArrayComputeInt, BatchAdd) {
    batch_add();
}

TEST_P(CArrayComputeLong, BatchAdd) {
    batch_add();
}

TEST_P(CArrayComputeDouble, BatchAdd) {
    batch_add();
}

TEST_P(CArrayComputeInt, BatchMul) {
    batch_mul();
}

TEST_P(CArrayComputeLong, BatchMul) {
    batch_mul();
}

TEST_P(CArrayComputeDouble, BatchMul) {
    batch_mul();
}

INSTANTIATE_TEST_SUITE_P(
    singlecore_compute,
    CArrayComputeInt,
    ::testing::ValuesIn(small_pow2),
    CArrayComputeInt::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
    singlecore_compute,
    CArrayComputeLong,
    ::testing::ValuesIn(small_pow2),
    CArrayComputeLong::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
    singlecore_compute,
    CArrayComputeDouble,
    ::testing::ValuesIn(small_pow2),
    CArrayComputeDouble::getTestCaseName
);
