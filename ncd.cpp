#include <pybind11/pybind11.h>
#include <iostream>
#include "gzip/compress.hpp"

namespace py = pybind11;

int get_compressed_length(std::string data) {
    const char* pointer = data.data();
    std::size_t size = data.size();

    std::string compressed = gzip::compress(pointer, size, 9);
    return static_cast<int>(compressed.size());
}

double compute_ncd(std::string a, std::string b) {
    int c_ab = get_compressed_length(a + b);
    int c_a = get_compressed_length(a);
    int c_b = get_compressed_length(b);

    return (double)(c_ab - std::min(c_a, c_b)) / (double)std::max(c_a, c_b);
}

PYBIND11_MODULE(ncd, m) {
    m.def("get_compressed_length", &get_compressed_length, "Get the length of a string after it was compressed by gzip.");
    m.def("compute_ncd", &compute_ncd, "Get the normalized compression distance of two strings.");
}