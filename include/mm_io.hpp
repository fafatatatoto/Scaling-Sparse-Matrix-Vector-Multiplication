#pragma once

#include "csr_matrix.hpp"

#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <stdexcept>
#include <string>
#include <vector>

namespace spmv {

enum class MtxSymmetry { General, Symmetric, SkewSymmetric, Hermitian };
enum class MtxField { Real, Integer, Pattern };

struct MtxData {
  int64_t nrows = 0;
  int64_t ncols = 0;
  // Entries declared in the file header, before expansion or coalescing.
  int64_t stored_nnz = 0;
  MtxSymmetry symmetry = MtxSymmetry::General;
  MtxField field = MtxField::Real;
  // Zero-based triplets after symmetric expansion. Duplicates are still here;
  // to_csr sums them.
  std::vector<int32_t> row;
  std::vector<int32_t> col;
  std::vector<double> value;
};

class MtxError : public std::runtime_error {
 public:
  MtxError(const std::string& name, int line, const std::string& message)
      : std::runtime_error(name + ":" + std::to_string(line) + ": " + message) {}
};

const char* to_string(MtxSymmetry symmetry);
const char* to_string(MtxField field);

// Reads a coordinate Matrix Market file.
// Symmetric, skew-symmetric, and real Hermitian files are expanded to a full
// triplet list. One triangle must be stored; storing both doubles the off-diagonals.
// Complex matrices and array format are rejected.
MtxData read_matrix_market(std::istream& in, const std::string& name);
MtxData read_matrix_market(const std::filesystem::path& path);

// Sorts, sums duplicate (row, col) entries, and builds CSR.
CsrMatrix to_csr(const MtxData& data);

CsrMatrix load_matrix_market(const std::filesystem::path& path);

}  // namespace spmv
