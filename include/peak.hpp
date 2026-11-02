#pragma once

#include <cstdint>

namespace spmv {

// Work in one call of peak_fma_block. The timed loop batches these calls.
inline constexpr int64_t k_peak_iters = 1 << 16;

// ISA-specific flop count of one peak_fma_block call. An FMA counts as 2 flops.
int64_t peak_flops_per_call();

const char* peak_isa_name();

// Register-only FMAs. The return value depends on every iteration.
double peak_fma_block();

}  // namespace spmv
