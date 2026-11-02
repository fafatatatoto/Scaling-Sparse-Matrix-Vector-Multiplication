#include "check.hpp"

#include "spmv.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace spmv {

SpmvCheck check_csr_against_mtx(const CsrMatrix& A, const MtxData& mtx) {
  if (A.nrows != mtx.nrows || A.ncols != mtx.ncols) {
    throw std::invalid_argument("CSR and Matrix Market dimensions differ");
  }
  if (mtx.row.size() != mtx.col.size() || mtx.row.size() != mtx.value.size()) {
    throw std::invalid_argument("triplet arrays have different lengths");
  }

  std::vector<double> x(static_cast<std::size_t>(A.ncols));
  for (int64_t col = 0; col < A.ncols; ++col) {
    x[static_cast<std::size_t>(col)] = 1.0 + 0.25 * static_cast<double>(col % 17);
  }

  std::vector<double> y_ref(static_cast<std::size_t>(A.nrows), 0.0);
  for (std::size_t k = 0; k < mtx.value.size(); ++k) {
    const std::size_t row = static_cast<std::size_t>(mtx.row[k]);
    const std::size_t col = static_cast<std::size_t>(mtx.col[k]);
    y_ref[row] += mtx.value[k] * x[col];
  }

  std::vector<double> y(static_cast<std::size_t>(A.nrows), 0.0);
  csr_spmv(A, x, y);

  SpmvCheck result;
  for (int64_t row = 0; row < A.nrows; ++row) {
    const double got = y[static_cast<std::size_t>(row)];
    const double ref = y_ref[static_cast<std::size_t>(row)];
    const double diff = std::abs(got - ref);
    result.max_abs_err = std::max(result.max_abs_err, diff);
    const double scale = std::max(1.0, std::abs(ref));
    result.max_rel_err = std::max(result.max_rel_err, diff / scale);
  }
  return result;
}

}  // namespace spmv
