#pragma once

#include <cstdint>
#include <vector>

namespace spmv {

// Serial CSR.
// col_idx is int32 so the traffic model stays at 4 bytes per nonzero.
// row_ptr is int64 so matrices with more than 2^31 nonzeros still load.
// nrows and ncols must fit in int32.
struct CsrMatrix {
  int64_t nrows = 0;
  int64_t ncols = 0;
  int64_t nnz = 0;
  std::vector<int64_t> row_ptr;
  std::vector<int32_t> col_idx;
  std::vector<double> values;
};

}  // namespace spmv
