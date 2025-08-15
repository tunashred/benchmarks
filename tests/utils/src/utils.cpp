#include <cstdlib>
#include <new>

#include "utils/utils.hpp"

void* safe_malloc(size_t size) {
    void* memory = malloc(size);
    if (memory == nullptr) {
        throw std::bad_alloc();
    }
    return memory;
}

// TODO: actually does free accept void* memory pointers?
void free_matrix(void**& matrix, size_t size) {
    for (size_t i = 0; i < size; i++) {
        free(matrix[i]);
    }
    free(matrix);
}
