#include "singlecore/iterate.hpp"
#include "utils/utils.hpp"
#include "utils/constants.hpp"

template <typename T>
void CArray<T>::SetUp() {
    size_t size = this->GetParam();
    size_t totalSize = size * sizeof *array;

    array = (T*) safe_malloc(totalSize);
    ASSERT_NE(array, nullptr) << "Unable to alloc array of size " << size;

    memset(array, 1, totalSize);
}

template <typename T>
void CArray<T>::TearDown() {
    free(array);
}

template <typename T>
void sequential_iterate(const CArray<T>* test) {
    size_t size = test->GetParam();
    for (size_t i = 0; i < LOOP_COUNT_200K; i++) {
        for (size_t j = 0; j < size; j++) {
            test->array[j]++;
        }
    }
}

template <typename T>
void reverse_sequential_iterate(const CArray<T>* test) {
    size_t size = test->GetParam();
    for (size_t i = 0; i < LOOP_COUNT_200K; i++) {
        for (size_t j = size - 1; j > 0; j--) {
            test->array[j]++;
        }
    }
}

template <typename T>
void jump_iterate(const CArray<T>* test) {
    constexpr size_t element_size = sizeof(T),
                     jump_size    = CACHE_LINE / element_size;
    
    size_t size = test->GetParam();
    
    for (size_t i = 0; i < LOOP_COUNT_200K; i++) {
        for (size_t k = 0; k < jump_size; k++) {
            size_t j = k;
            while (j < size) {
                test->array[j]++;
        
                j += jump_size;
            }
        }
    }
}

template <typename T>
void reverse_jump_iterate(const CArray<T>* test) {
    constexpr size_t element_size = sizeof(T),
                     jump_size    = CACHE_LINE / element_size;
    
    size_t size = test->GetParam();

    for (size_t i = 0; i < LOOP_COUNT_200K; i++) {
        for (size_t k = 0; k < jump_size; k++) {
            size_t j = size - k - 1;
            while (j < size) {
                test->array[j]++;
        
                j -= jump_size;
            }
        }
    }
}

using CArrayInt = CArray<int>;
using CArrayLong = CArray<long>;
using CArrayDouble = CArray<double>;

TEST_P(CArrayInt, SequentialIterate) {
    sequential_iterate(this);
}

TEST_P(CArrayLong, SequentialIterate) {
    sequential_iterate(this);
}

TEST_P(CArrayDouble, SequentialIterate) {
    sequential_iterate(this);
}

TEST_P(CArrayInt, DISABLED_ReverseSequentialIterate) {
    reverse_sequential_iterate(this);
}

TEST_P(CArrayLong, DISABLED_ReverseSequentialIterate) {
    reverse_sequential_iterate(this);
}

TEST_P(CArrayDouble, DISABLED_ReverseSequentialIterate) {
    reverse_sequential_iterate(this);
}

TEST_P(CArrayInt, DISABLED_JumpIterate) {
    jump_iterate(this);
}

TEST_P(CArrayLong, DISABLED_JumpIterate) {
    jump_iterate(this);
}

TEST_P(CArrayDouble, DISABLED_JumpIterate) {
    jump_iterate(this);
}

TEST_P(CArrayInt, DISABLED_ReverseJumpIterate) {
    reverse_jump_iterate(this);
}

TEST_P(CArrayLong, DISABLED_ReverseJumpIterate) {
    reverse_jump_iterate(this);
}

TEST_P(CArrayDouble, DISABLED_ReverseJumpIterate) {
    reverse_jump_iterate(this);
}

INSTANTIATE_TEST_SUITE_P(
    scalar_singlecore_caching,
    CArrayInt,
    ::testing::Values(10, 100, 1000, 10000, 100000, 1000000, 2000000, 3000000),
    CArrayInt::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
    scalar_singlecore_caching,
    CArrayLong,
    ::testing::Values(10, 100, 1000, 10000, 100000, 1000000, 2000000, 3000000),
    CArrayLong::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
    scalar_singlecore_caching,
    CArrayDouble,
    ::testing::Values(10, 100, 1000, 10000, 100000, 1000000, 2000000, 3000000),
    CArrayDouble::getTestCaseName
);
