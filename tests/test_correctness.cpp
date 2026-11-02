#include "benchmark.hpp"
#include "check.hpp"
#include "dense.hpp"
#include "mm_io.hpp"
#include "source_dir.hpp"
#include "spmv.hpp"
#include "traffic.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

int g_failed = 0;

void fail(const std::string& message) {
  std::cerr << message << '\n';
  ++g_failed;
}

void expect(bool condition, const char* expression, const char* file, int line) {
  if (!condition) {
    fail(std::string(file) + ":" + std::to_string(line) + " failed: " + expression);
  }
}

void expect_close(double got, double want, const char* file, int line) {
  const double scale = std::max(1.0, std::max(std::abs(got), std::abs(want)));
  if (std::abs(got - want) > 1e-9 * scale) {
    fail(std::string(file) + ":" + std::to_string(line) + " got " + std::to_string(got) + " want " +
         std::to_string(want));
  }
}

#define EXPECT(cond) expect((cond), #cond, __FILE__, __LINE__)
#define EXPECT_CLOSE(got, want) expect_close((got), (want), __FILE__, __LINE__)

std::string fixture(const char* name) {
  return std::string(SPMV_SOURCE_DIR) + "/benchmarks/" + name;
}

std::vector<double> multiply(const spmv::CsrMatrix& matrix, const std::vector<double>& x) {
  std::vector<double> y(static_cast<std::size_t>(matrix.nrows), 0.0);
  spmv::csr_spmv(matrix, x, y);
  return y;
}

spmv::CsrMatrix load_fixture(const char* name, spmv::MtxData* mtx = nullptr) {
  spmv::MtxData data = spmv::read_matrix_market(fixture(name));
  spmv::CsrMatrix matrix = spmv::to_csr(data);
  if (mtx != nullptr) {
    *mtx = std::move(data);
  }
  return matrix;
}

spmv::MtxData parse_text(const char* name, const std::string& text) {
  std::istringstream in(text);
  return spmv::read_matrix_market(in, name);
}

template <class F>
void expect_error(const char* label, F&& function) {
  try {
    function();
    fail(std::string(label) + ": expected MtxError");
  } catch (const spmv::MtxError&) {
  }
}

void test_tiny() {
  spmv::MtxData mtx;
  const spmv::CsrMatrix matrix = load_fixture("tiny.mtx", &mtx);
  EXPECT(matrix.nrows == 4);
  EXPECT(matrix.ncols == 4);
  EXPECT(mtx.stored_nnz == 6);
  EXPECT(mtx.row.size() == 6);
  EXPECT(matrix.nnz == 6);
  EXPECT(matrix.row_ptr.size() == 5);
  EXPECT(matrix.row_ptr[0] == 0);
  EXPECT(matrix.row_ptr[1] == 2);
  EXPECT(matrix.row_ptr[2] == 3);
  EXPECT(matrix.row_ptr[3] == 5);
  EXPECT(matrix.row_ptr[4] == 6);
  EXPECT(matrix.col_idx[0] == 0);
  EXPECT(matrix.col_idx[1] == 1);
  EXPECT(matrix.values[0] == 1.0);
  EXPECT(matrix.values[1] == 2.0);

  const std::vector<double> y = multiply(matrix, {1, 2, 3, 4});
  EXPECT_CLOSE(y[0], 5);
  EXPECT_CLOSE(y[1], 6);
  EXPECT_CLOSE(y[2], 24);
  EXPECT_CLOSE(y[3], 24);

  const spmv::TrafficEstimate traffic = spmv::estimate_traffic(matrix);
  EXPECT(traffic.unique_cols == 3);
  EXPECT(traffic.bytes_x_reused == 200);
  EXPECT(traffic.bytes_x_reloaded == 224);

  const spmv::SpmvCheck compared = spmv::check_csr_against_mtx(matrix, mtx);
  EXPECT(compared.max_rel_err < 1e-12);
}

void test_symmetric() {
  spmv::MtxData mtx;
  const spmv::CsrMatrix matrix = load_fixture("tiny_symmetric.mtx", &mtx);
  EXPECT(mtx.symmetry == spmv::MtxSymmetry::Symmetric);
  EXPECT(mtx.stored_nnz == 4);
  EXPECT(mtx.row.size() == 6);
  EXPECT(matrix.nnz == 6);
  const std::vector<double> y = multiply(matrix, {1, 2, 3});
  EXPECT_CLOSE(y[0], 5);
  EXPECT_CLOSE(y[1], 20);
  EXPECT_CLOSE(y[2], 8);
}

void test_upper_triangle() {
  const spmv::MtxData mtx = parse_text("upper",
                                       "%%MatrixMarket matrix coordinate real symmetric\n"
                                       "2 2 2\n"
                                       "1 1 1.0\n"
                                       "1 2 3.0\n");
  const spmv::CsrMatrix matrix = spmv::to_csr(mtx);
  EXPECT(matrix.nnz == 3);
  const std::vector<double> y = multiply(matrix, {1, 2});
  EXPECT_CLOSE(y[0], 7);
  EXPECT_CLOSE(y[1], 3);
}

void test_skew() {
  const spmv::CsrMatrix matrix = load_fixture("tiny_skew.mtx");
  EXPECT(matrix.nnz == 4);
  const std::vector<double> y = multiply(matrix, {1, 2, 3});
  EXPECT_CLOSE(y[0], 3);
  EXPECT_CLOSE(y[1], 1.5);
  EXPECT_CLOSE(y[2], -2);
}

void test_skew_zero_diagonal_is_dropped() {
  const spmv::MtxData mtx = parse_text("skew0",
                                       "%%MatrixMarket matrix coordinate real skew-symmetric\n"
                                       "2 2 2\n"
                                       "1 1 0.0\n"
                                       "2 1 4.0\n");
  const spmv::CsrMatrix matrix = spmv::to_csr(mtx);
  EXPECT(mtx.stored_nnz == 2);
  EXPECT(matrix.nnz == 2);
  const std::vector<double> y = multiply(matrix, {1, 2});
  EXPECT_CLOSE(y[0], -8);
  EXPECT_CLOSE(y[1], 4);
}

void test_pattern() {
  const spmv::CsrMatrix matrix = load_fixture("tiny_pattern.mtx");
  EXPECT(matrix.nnz == 2);
  EXPECT(matrix.values[0] == 1.0);
  EXPECT(matrix.values[1] == 1.0);
  const std::vector<double> y = multiply(matrix, {1, 2});
  EXPECT_CLOSE(y[0], 2);
  EXPECT_CLOSE(y[1], 1);
}

void test_duplicates() {
  spmv::MtxData mtx;
  const spmv::CsrMatrix matrix = load_fixture("tiny_duplicates.mtx", &mtx);
  EXPECT(mtx.row.size() == 4);
  EXPECT(matrix.nnz == 3);
  const std::vector<double> y = multiply(matrix, {1, 2});
  EXPECT_CLOSE(y[0], 11);
  EXPECT_CLOSE(y[1], 3);
  const spmv::SpmvCheck compared = spmv::check_csr_against_mtx(matrix, mtx);
  EXPECT(compared.max_rel_err < 1e-12);
}

void test_rectangular() {
  const spmv::CsrMatrix matrix = load_fixture("tiny_rectangular.mtx");
  EXPECT(matrix.nrows == 2);
  EXPECT(matrix.ncols == 3);
  const std::vector<double> y = multiply(matrix, {1, 2, 3});
  EXPECT_CLOSE(y[0], 7);
  EXPECT_CLOSE(y[1], 10);
}

void test_integer_and_fortran_exponent() {
  const spmv::CsrMatrix matrix = load_fixture("tiny_integer.mtx");
  const std::vector<double> y = multiply(matrix, {1, 2});
  EXPECT_CLOSE(y[0], 6);
  EXPECT_CLOSE(y[1], -4);

  const spmv::MtxData mtx = parse_text("fortran",
                                       "%%MatrixMarket matrix coordinate real general\n"
                                       "1 1 1\n"
                                       "1 1 2.5D+01\n");
  const spmv::CsrMatrix fortran = spmv::to_csr(mtx);
  const std::vector<double> y_fortran = multiply(fortran, {1});
  EXPECT_CLOSE(y_fortran[0], 25);
}

void test_explicit_zero_is_kept() {
  const spmv::MtxData mtx = parse_text("zero",
                                       "%%MatrixMarket matrix coordinate real general\n"
                                       "1 1 1\n"
                                       "1 1 0.0\n");
  const spmv::CsrMatrix matrix = spmv::to_csr(mtx);
  EXPECT(matrix.nnz == 1);
  EXPECT(matrix.values[0] == 0.0);
}

void test_hermitian_real() {
  const spmv::MtxData mtx = parse_text("herm",
                                       "%%MatrixMarket matrix coordinate real hermitian\n"
                                       "2 2 2\n"
                                       "1 1 4.0\n"
                                       "2 1 5.0\n");
  const spmv::CsrMatrix matrix = spmv::to_csr(mtx);
  EXPECT(matrix.nnz == 3);
  const std::vector<double> y = multiply(matrix, {1, 2});
  EXPECT_CLOSE(y[0], 4 + 10);
  EXPECT_CLOSE(y[1], 5);
}

void test_rejections() {
  expect_error("complex", [] {
    parse_text("complex",
               "%%MatrixMarket matrix coordinate complex general\n"
               "1 1 1\n"
               "1 1 1.0 2.0\n");
  });
  expect_error("array", [] {
    parse_text("array",
               "%%MatrixMarket matrix array real general\n"
               "1 1\n"
               "1.0\n");
  });
  expect_error("range", [] {
    parse_text("range",
               "%%MatrixMarket matrix coordinate real general\n"
               "2 2 1\n"
               "3 1 1.0\n");
  });
  expect_error("truncated", [] {
    parse_text("truncated",
               "%%MatrixMarket matrix coordinate real general\n"
               "2 2 2\n"
               "1 1 1.0\n");
  });
  expect_error("skew diagonal", [] {
    parse_text("skewdiag",
               "%%MatrixMarket matrix coordinate real skew-symmetric\n"
               "1 1 1\n"
               "1 1 1.0\n");
  });
  expect_error("missing", [] { spmv::read_matrix_market(fixture("does_not_exist.mtx")); });
}

void test_dense() {
  const spmv::DenseCost small = spmv::dense_cost(4);
  EXPECT(small.flops == 32.0);
  EXPECT(small.bytes == 224.0);
  const spmv::DenseMatrix matrix = spmv::make_dense_ones(4);
  std::vector<double> x(4, 1.0);
  std::vector<double> y(4, 0.0);
  spmv::dense_matvec(matrix.a.data(), x.data(), y.data(), 4);
  for (double value : y) {
    EXPECT_CLOSE(value, 4.0);
  }
  const spmv::DenseCost streamed = spmv::dense_cost(4096);
  EXPECT(streamed.ai > 0.24);
  EXPECT(streamed.ai < 0.25);
}

void test_benchmark_formula() {
  const spmv::CsrMatrix matrix = load_fixture("tiny.mtx");
  spmv::BenchConfig config;
  config.warmup_batches = 0;
  config.min_samples = 3;
  config.max_samples = 5;
  config.min_seconds = 0.05;
  config.target_sample_seconds = 0.01;

  const spmv::BenchResult result = spmv::run_benchmark(matrix, config);
  EXPECT(result.samples >= 3);
  EXPECT(result.samples <= 5);
  EXPECT(result.batch >= 1);
  EXPECT(result.median_s > 0.0);
  EXPECT(result.min_s <= result.median_s);
  EXPECT(result.median_s <= result.max_s);
  EXPECT(result.bytes_x_reused == 200);
  EXPECT(result.bytes_x_reloaded == 224);
  EXPECT(result.unique_cols == 3);

  const double gflops = 2.0 * static_cast<double>(matrix.nnz) / result.median_s / 1e9;
  EXPECT_CLOSE(result.gflops, gflops);
  EXPECT_CLOSE(result.ai_x_reused, 12.0 / 200.0);
  EXPECT_CLOSE(result.ai_x_reloaded, 12.0 / 224.0);
}

}  // namespace

int main() {
  test_tiny();
  test_symmetric();
  test_upper_triangle();
  test_skew();
  test_skew_zero_diagonal_is_dropped();
  test_pattern();
  test_duplicates();
  test_rectangular();
  test_integer_and_fortran_exponent();
  test_explicit_zero_is_kept();
  test_hermitian_real();
  test_rejections();
  test_dense();
  test_benchmark_formula();

  if (g_failed != 0) {
    std::cerr << g_failed << " check(s) failed\n";
    return 1;
  }
  std::cout << "ok\n";
  return 0;
}
