#pragma once

#include "csr_matrix.hpp"

#include <span>

namespace spmv {

// Serial CSR SpMV: y = A * x.
// Requires x.size() == A.ncols and y.size() == A.nrows.
// Overwrites y. This is not y += Ax.
void csr_spmv(const CsrMatrix& A, std::span<const double> x, std::span<double> y);

}  // namespace spmv
