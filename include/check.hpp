#pragma once

#include "csr_matrix.hpp"
#include "mm_io.hpp"

namespace spmv {

struct SpmvCheck {
  double max_abs_err = 0;
  double max_rel_err = 0;
};

// Independent reference: accumulate the expanded triplets.
// x[j] = 1 + 0.25 * (j % 17), so a wrong column index fails the check.
// An all-ones x would not, because every column then contributes the same factor.
SpmvCheck check_csr_against_mtx(const CsrMatrix& A, const MtxData& mtx);

}  // namespace spmv
