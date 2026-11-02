#pragma once

#include "benchmark.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>
#include <vector>

namespace spmv {
namespace timed {

struct Samples {
  double median_s = 0;
  double min_s = 0;
  double max_s = 0;
  int samples = 0;
  int batch = 0;
};

inline double elapsed_seconds(std::chrono::steady_clock::time_point start,
                              std::chrono::steady_clock::time_point stop) {
  return std::chrono::duration<double>(stop - start).count();
}

inline void validate_config(const BenchConfig& config) {
  if (config.warmup_batches < 0) {
    throw std::invalid_argument("warmup_batches must be >= 0");
  }
  if (config.min_samples < 1) {
    throw std::invalid_argument("min_samples must be >= 1");
  }
  if (config.max_samples < config.min_samples) {
    throw std::invalid_argument("max_samples must be >= min_samples");
  }
  if (config.min_seconds < 0.0) {
    throw std::invalid_argument("min_seconds must be >= 0");
  }
  if (!(config.target_sample_seconds > 0.0)) {
    throw std::invalid_argument("sample seconds must be > 0");
  }
}

// Times repeated calls of once(). once() must perform the kernel and keep a
// live side effect; the kernel itself stays in another translation unit.
template <class F>
Samples run(const BenchConfig& config, F&& once) {
  validate_config(config);
  once();
  for (int i = 0; i < config.warmup_batches; ++i) {
    once();
  }

  constexpr int k_max_batch = 1 << 20;
  auto time_batch = [&](int batch) {
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < batch; ++i) {
      once();
    }
    return elapsed_seconds(start, std::chrono::steady_clock::now());
  };

  int batch = 1;
  while (batch < k_max_batch) {
    if (time_batch(batch) >= config.target_sample_seconds) {
      break;
    }
    batch *= 2;
  }

  for (int i = 0; i < config.warmup_batches; ++i) {
    time_batch(batch);
  }

  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(config.max_samples));
  double total = 0.0;
  while (static_cast<int>(samples.size()) < config.max_samples &&
         (static_cast<int>(samples.size()) < config.min_samples || total < config.min_seconds)) {
    const double sample = time_batch(batch);
    if (sample <= 0.0) {
      throw std::runtime_error("timer did not advance; the batch is too small for this clock");
    }
    samples.push_back(sample / static_cast<double>(batch));
    total += sample;
  }

  std::sort(samples.begin(), samples.end());
  const std::size_t n = samples.size();
  Samples out;
  out.min_s = samples.front();
  out.max_s = samples.back();
  out.median_s = (n % 2 == 1) ? samples[n / 2] : 0.5 * (samples[n / 2 - 1] + samples[n / 2]);
  out.samples = static_cast<int>(n);
  out.batch = batch;
  return out;
}

}  // namespace timed
}  // namespace spmv
