#pragma once

#include "csr_matrix.hpp"

#include <cstdint>

namespace spmv {

// Model traffic for one SpMV. See docs/methodology.md.
// bytes_x_reused: each distinct column index loads x once.
// bytes_x_reloaded: every nonzero reloads x.
struct TrafficEstimate {
  double bytes_x_reused = 0;
  double bytes_x_reloaded = 0;
  int64_t unique_cols = 0;
};

TrafficEstimate estimate_traffic(const CsrMatrix& A);

}  // namespace spmv
