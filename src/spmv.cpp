#include "spmv.hpp"

#include <stdexcept>

namespace spmv {

void csr_spmv(const CsrMatrix& A, std::span<const double> x, std::span<double> y) {
  if (x.size() != static_cast<std::size_t>(A.ncols) || y.size() != static_cast<std::size_t>(A.nrows)) {
    throw std::invalid_argument("csr_spmv: x and y dimensions do not match A");
  }

  const int64_t nrows = A.nrows;
  const int64_t* row_ptr = A.row_ptr.data();
  const int32_t* col_idx = A.col_idx.data();
  const double* values = A.values.data();
  const double* x_data = x.data();
  double* y_data = y.data();

  for (int64_t row = 0; row < nrows; ++row) {
    double sum = 0.0;
    const int64_t begin = row_ptr[row];
    const int64_t end = row_ptr[row + 1];
    for (int64_t k = begin; k < end; ++k) {
      sum += values[k] * x_data[col_idx[k]];
    }
    y_data[row] = sum;
  }
}

void csr_spmv_omp(const CsrMatrix& A, std::span<const double> x, std::span<double> y) {
  if (x.size() != static_cast<std::size_t>(A.ncols) || y.size() != static_cast<std::size_t>(A.nrows)) {
    throw std::invalid_argument("csr_spmv: x and y dimensions do not match A");
  }

  const int64_t nrows = A.nrows;
  const int64_t* row_ptr = A.row_ptr.data();
  const int32_t* col_idx = A.col_idx.data();
  const double* values = A.values.data();
  const double* x_data = x.data();
  double* y_data = y.data();

  for (int64_t row = 0; row < nrows; ++row) {
    double sum = 0.0;
    const int64_t begin = row_ptr[row];
    const int64_t end = row_ptr[row + 1];
    for (int64_t k = begin; k < end; ++k) {
      sum += values[k] * x_data[col_idx[k]];
    }
    y_data[row] = sum;
  }
}


}  // namespace spmv
