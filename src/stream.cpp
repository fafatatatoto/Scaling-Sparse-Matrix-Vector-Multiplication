#include "stream.hpp"

namespace spmv {

void stream_triad(double* a, const double* b, const double* c, int64_t n, double scalar) {
  for (int64_t i = 0; i < n; ++i) {
    a[i] = b[i] + scalar * c[i];
  }
}

}  // namespace spmv
