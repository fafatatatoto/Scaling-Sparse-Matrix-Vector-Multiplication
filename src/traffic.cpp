#include "traffic.hpp"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <vector>

namespace spmv {

TrafficEstimate estimate_traffic(const CsrMatrix& A) {
  static_assert(sizeof(double) == 8, "traffic model assumes 8-byte doubles");
  static_assert(sizeof(int32_t) == 4, "traffic model assumes 4-byte column indices");
  static_assert(sizeof(int64_t) == 8, "traffic model assumes 8-byte row pointers");

  std::vector<int32_t> columns = A.col_idx;
  std::sort(columns.begin(), columns.end());
  const auto unique_end = std::unique(columns.begin(), columns.end());
  const int64_t unique_cols = static_cast<int64_t>(std::distance(columns.begin(), unique_end));

  const double values_bytes = 8.0 * static_cast<double>(A.nnz);
  const double col_bytes = 4.0 * static_cast<double>(A.nnz);
  const double row_ptr_bytes = 8.0 * static_cast<double>(A.nrows + 1);
  const double y_bytes = 16.0 * static_cast<double>(A.nrows);
  const double matrix_and_y = values_bytes + col_bytes + row_ptr_bytes + y_bytes;

  TrafficEstimate estimate;
  estimate.unique_cols = unique_cols;
  estimate.bytes_x_reused = matrix_and_y + 8.0 * static_cast<double>(unique_cols);
  estimate.bytes_x_reloaded = matrix_and_y + 8.0 * static_cast<double>(A.nnz);
  return estimate;
}

}  // namespace spmv
