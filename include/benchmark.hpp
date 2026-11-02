#pragma once

#include "csr_matrix.hpp"

#include <cstdint>

namespace spmv {

struct BenchConfig {
  int warmup_batches = 2;
  int min_samples = 7;
  int max_samples = 100;
  double min_seconds = 1.0;
  double target_sample_seconds = 0.01;
};

struct BenchResult {
  double median_s = 0;
  double min_s = 0;
  double max_s = 0;
  int samples = 0;
  int batch = 0;
  double gflops = 0;
  double bytes_x_reused = 0;
  double bytes_x_reloaded = 0;
  double gbps_x_reused = 0;
  double gbps_x_reloaded = 0;
  double ai_x_reused = 0;
  double ai_x_reloaded = 0;
  int64_t unique_cols = 0;
};

// Times y = A*x with x = 1. Reports the median per-call time.
// Bandwidth and arithmetic intensity use estimate_traffic; they are not
// hardware-counter measurements.
BenchResult run_benchmark(const CsrMatrix& A, const BenchConfig& config = {});

// Times dense y = A*x, A filled with ones. Both byte-model columns describe
// the same traffic: A streams once and x is read once.
BenchResult run_dense_benchmark(int64_t n, const BenchConfig& config = {});

const char* build_flags();
const char* compiler_id();

}  // namespace spmv
