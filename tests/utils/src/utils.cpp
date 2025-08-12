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
