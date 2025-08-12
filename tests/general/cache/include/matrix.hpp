#pragma once

#include <gtest/gtest.h>

constexpr size_t BLOCK_SIZE = 64;

template <typename T>
class CMatrix : public testing::TestWithParam<size_t> {
protected:
    void SetUp() override;

    void TearDown() override;
public:
    T** matrix_A;
    T** matrix_B;
    T** matrix_C;

    void naive_mul();

    void optimized_mul();

    void block_mul();

    static std::string getTestCaseName(const ::testing::TestParamInfo<size_t>& info) {
        return "size_" + std::to_string(info.param) + "x" + std::to_string(info.param);
    }
};

template <typename T>
void create_matrix(T**& matrix, size_t size);

void free_matrix(void**& matrix, size_t size);
