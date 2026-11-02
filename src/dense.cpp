#include "dense.hpp"

#include <stdexcept>

namespace spmv {

DenseMatrix make_dense_ones(int64_t n) {
  if (n < 1 || n > 16384) {
    throw std::invalid_argument("dense n must be in 1..16384 so the matrix stays within 2 GiB");
  }
  DenseMatrix matrix;
  matrix.n = n;
  matrix.a.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 1.0);
  return matrix;
}

void dense_matvec(const double* a, const double* x, double* y, int64_t n) {
  for (int64_t row = 0; row < n; ++row) {
    const double* values = a + static_cast<std::size_t>(row) * static_cast<std::size_t>(n);
    double sum = 0.0;
    for (int64_t col = 0; col < n; ++col) {
      sum += values[col] * x[col];
    }
    y[row] = sum;
  }
}

DenseCost dense_cost(int64_t n) {
  const double nn = static_cast<double>(n) * static_cast<double>(n);
  DenseCost cost;
  cost.flops = 2.0 * nn;
  cost.bytes = 8.0 * nn + 24.0 * static_cast<double>(n);
  cost.ai = cost.bytes > 0.0 ? cost.flops / cost.bytes : 0.0;
  return cost;
}

}  // namespace spmv
