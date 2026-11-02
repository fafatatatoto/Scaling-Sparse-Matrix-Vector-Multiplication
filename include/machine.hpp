#pragma once

#include "benchmark.hpp"

#include <cstdint>
#include <string>

namespace spmv {

// Single-thread roofs for the serial roofline.
// peak_gflops is a register FMA microbenchmark, not a vendor spec.
// stream_gbps is a triad using the STREAM convention of 24 bytes per element.
struct MachineRoof {
  std::string cpu;
  int64_t llc_bytes = 0;
  double peak_gflops = 0;
  double stream_gbps = 0;
  int64_t stream_n = 0;
  double stream_bytes = 0;
  const char* peak_isa = "";
  double peak_median_s = 0;
  double stream_median_s = 0;
};

MachineRoof measure_machine(const BenchConfig& config = {});

}  // namespace spmv
