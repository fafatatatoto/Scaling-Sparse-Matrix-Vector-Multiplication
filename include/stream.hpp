#pragma once

#include <cstdint>

namespace spmv {

// a[i] = b[i] + scalar * c[i]. STREAM counts 24 bytes per element (two reads
// and one write), without a write-allocate.
inline constexpr int64_t k_stream_bytes_per_element = 24;

void stream_triad(double* a, const double* b, const double* c, int64_t n, double scalar);

}  // namespace spmv
