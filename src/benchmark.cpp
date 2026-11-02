#include "benchmark.hpp"

#include "dense.hpp"
#include "spmv.hpp"
#include "timed_loop.hpp"
#include "traffic.hpp"

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

namespace spmv {
namespace {

// Observed by the timed loop so a compiler cannot delete the kernel as a dead store.
// Link-time optimization stays off (see CMakeLists.txt).
volatile double g_observed = 0;

void fill_result(BenchResult& result, const timed::Samples& samples, double flops, double bytes_reused,
                 double bytes_reloaded, int64_t unique_cols) {
  result.median_s = samples.median_s;
  result.min_s = samples.min_s;
  result.max_s = samples.max_s;
  result.samples = samples.samples;
  result.batch = samples.batch;
  result.bytes_x_reused = bytes_reused;
  result.bytes_x_reloaded = bytes_reloaded;
  result.unique_cols = unique_cols;
  if (samples.median_s > 0.0) {
    result.gflops = flops / samples.median_s / 1e9;
    result.gbps_x_reused = bytes_reused / samples.median_s / 1e9;
    result.gbps_x_reloaded = bytes_reloaded / samples.median_s / 1e9;
  }
  if (bytes_reused > 0.0) {
    result.ai_x_reused = flops / bytes_reused;
  }
  if (bytes_reloaded > 0.0) {
    result.ai_x_reloaded = flops / bytes_reloaded;
  }
}

}  // namespace

const char* build_flags() {
#ifdef NDEBUG
  return "-O3 -march=native";
#else
  return "-O0 -g";
#endif
}

const char* compiler_id() {
#if defined(__clang__)
  return "clang " __clang_version__;
#elif defined(__GNUC__)
  return "gcc " __VERSION__;
#else
  return "unknown";
#endif
}

BenchResult run_benchmark(const CsrMatrix& A, const BenchConfig& config) {
  std::vector<double> x(static_cast<std::size_t>(A.ncols), 1.0);
  std::vector<double> y(static_cast<std::size_t>(A.nrows), 0.0);
  const std::span<const double> x_span(x);
  const std::span<double> y_span(y);

  const timed::Samples samples = timed::run(config, [&] {
    csr_spmv(A, x_span, y_span);
    if (!y.empty()) {
      g_observed = y[0];
    }
  });

  const TrafficEstimate traffic = estimate_traffic(A);
  BenchResult result;
  fill_result(result, samples, 2.0 * static_cast<double>(A.nnz), traffic.bytes_x_reused,
              traffic.bytes_x_reloaded, traffic.unique_cols);
  return result;
}

BenchResult run_dense_benchmark(int64_t n, const BenchConfig& config) {
  const DenseMatrix matrix = make_dense_ones(n);
  std::vector<double> x(static_cast<std::size_t>(n), 1.0);
  std::vector<double> y(static_cast<std::size_t>(n), 0.0);

  dense_matvec(matrix.a.data(), x.data(), y.data(), n);
  double max_abs = 0.0;
  for (double value : y) {
    max_abs = std::max(max_abs, std::abs(value - static_cast<double>(n)));
  }
  if (max_abs != 0.0) {
    throw std::runtime_error("dense matvec failed the all-ones check");
  }

  const timed::Samples samples = timed::run(config, [&] {
    dense_matvec(matrix.a.data(), x.data(), y.data(), n);
    g_observed = y[0];
  });

  const DenseCost cost = dense_cost(n);
  BenchResult result;
  fill_result(result, samples, cost.flops, cost.bytes, cost.bytes, n);
  return result;
}

}  // namespace spmv
