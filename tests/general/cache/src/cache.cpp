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
void iterate(const CArray<T>* test) {
    size_t size = test->GetParam();
    for (size_t i = 0; i < 200000; i++) {
        for (size_t j = 0; j < size; j++) {
            test->array[j]++;
        }
    }
}

TEST_P(CArrayInt, Iterate) {
    iterate(this);
}

TEST_P(CArrayLong, Iterate) {
    iterate(this);
}

TEST_P(CArrayFloat, Iterate) {
    iterate(this);
}

TEST_P(CArrayDouble, Iterate) {
    iterate(this);
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
