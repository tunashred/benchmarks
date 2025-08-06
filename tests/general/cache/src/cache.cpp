#include "cache.hpp"

template <typename T>
void CArray<T>::SetUp() {
    size_t size = this->GetParam();
    size_t totalSize = size * sizeof *array;

    array = (T*) malloc(totalSize);
    ASSERT_NE(array, nullptr) << "Unable to alloc array of size " << size;

    // warming up the memory
    memset(array, 1, totalSize);
}

template <typename T>
void CArray<T>::TearDown() {
    free(array);
}

using CArrayInt = CArray<int>;
using CArrayLong = CArray<long>;
using CArrayFloat = CArray<float>;
using CArrayDouble = CArray<double>;

template <typename T>
void sequential_iterate(const CArray<T>* test) {
    size_t size = test->GetParam();
    for (size_t i = 0; i < 200000; i++) {
        for (size_t j = 0; j < size; j++) {
            test->array[j]++;
        }
    }
}

template <typename T>
void jump_iterate(const CArray<T>* test) {
    constexpr size_t element_size = sizeof(T),
                     jump_size    = CACHE_LINE / element_size;
    
    size_t size = test->GetParam();
    
    for (size_t i = 0; i < 200000; i++) {
        for (size_t k = 0; k < jump_size; k++) {
            size_t j = k;
            while (j < size) {
                test->array[j]++;
        
                j += jump_size;
            }
        }
    }
}

TEST_P(CArrayInt, SequentialIterate) {
    sequential_iterate(this);
}

TEST_P(CArrayLong, SequentialIterate) {
    sequential_iterate(this);
}

TEST_P(CArrayFloat, SequentialIterate) {
    sequential_iterate(this);
}

TEST_P(CArrayDouble, SequentialIterate) {
    sequential_iterate(this);
}

TEST_P(CArrayInt, JumpIterate) {
    jump_iterate(this);
}

TEST_P(CArrayLong, JumpIterate) {
    jump_iterate(this);
}

TEST_P(CArrayFloat, JumpIterate) {
    jump_iterate(this);
}

TEST_P(CArrayDouble, JumpIterate) {
    jump_iterate(this);
}

INSTANTIATE_TEST_SUITE_P(
    GeneralCaching,
    CArrayInt,
    ::testing::Values(10, 100, 1000, 10000, 100000, 1000000, 2000000),
    getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
    GeneralCaching,
    CArrayLong,
    ::testing::Values(10, 100, 1000, 10000, 100000, 1000000, 2000000),
    getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
    GeneralCaching,
    CArrayFloat,
    ::testing::Values(10, 100, 1000, 10000, 100000, 1000000, 2000000),
    getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
    GeneralCaching,
    CArrayDouble,
    ::testing::Values(10, 100, 1000, 10000, 100000, 1000000, 2000000),
    getTestCaseName
);
