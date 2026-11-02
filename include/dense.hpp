#pragma once

#include <cstdint>
#include <vector>

namespace spmv {

// Row-major dense matrix used as the arithmetic-intensity control for SpMV.
struct DenseMatrix {
  int64_t n = 0;
  std::vector<double> a;
};

struct DenseCost {
  double bytes = 0;
  double flops = 0;
  double ai = 0;
};

// A is filled with 1, so y = A * 1 is the vector of n's.
// n * n * 8 bytes must stay within 2 GiB (n <= 16384).
DenseMatrix make_dense_ones(int64_t n);

// y = A * x. A is row-major n*n. x and y have length n.
void dense_matvec(const double* a, const double* x, double* y, int64_t n);

// A is read once, x is read once, y is write-allocated.
// For the n we benchmark, x fits in cache while A streams from DRAM.
DenseCost dense_cost(int64_t n);

}  // namespace spmv
