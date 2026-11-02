#include "peak.hpp"

#if defined(__AVX512F__) || defined(__AVX__)
#include <immintrin.h>
#endif

#include <cstddef>

namespace spmv {
namespace {

#if defined(__AVX512F__)

double peak_fma_block_impl() {
  __m512d a0 = _mm512_set1_pd(1.0);
  __m512d a1 = _mm512_set1_pd(1.1);
  __m512d a2 = _mm512_set1_pd(1.2);
  __m512d a3 = _mm512_set1_pd(1.3);
  __m512d a4 = _mm512_set1_pd(1.4);
  __m512d a5 = _mm512_set1_pd(1.5);
  __m512d a6 = _mm512_set1_pd(1.6);
  __m512d a7 = _mm512_set1_pd(1.7);
  const __m512d b = _mm512_set1_pd(1.0000001);
  const __m512d c = _mm512_set1_pd(0.9999999);
  for (int64_t i = 0; i < k_peak_iters; ++i) {
    a0 = _mm512_fmadd_pd(a0, b, c);
    a1 = _mm512_fmadd_pd(a1, b, c);
    a2 = _mm512_fmadd_pd(a2, b, c);
    a3 = _mm512_fmadd_pd(a3, b, c);
    a4 = _mm512_fmadd_pd(a4, b, c);
    a5 = _mm512_fmadd_pd(a5, b, c);
    a6 = _mm512_fmadd_pd(a6, b, c);
    a7 = _mm512_fmadd_pd(a7, b, c);
  }
  __m512d sum = _mm512_add_pd(a0, a1);
  sum = _mm512_add_pd(sum, a2);
  sum = _mm512_add_pd(sum, a3);
  sum = _mm512_add_pd(sum, a4);
  sum = _mm512_add_pd(sum, a5);
  sum = _mm512_add_pd(sum, a6);
  sum = _mm512_add_pd(sum, a7);
  alignas(64) double lanes[8];
  _mm512_store_pd(lanes, sum);
  double total = 0.0;
  for (double lane : lanes) {
    total += lane;
  }
  return total;
}

#elif defined(__AVX__) && defined(__FMA__)

double peak_fma_block_impl() {
  __m256d a0 = _mm256_set1_pd(1.0);
  __m256d a1 = _mm256_set1_pd(1.1);
  __m256d a2 = _mm256_set1_pd(1.2);
  __m256d a3 = _mm256_set1_pd(1.3);
  __m256d a4 = _mm256_set1_pd(1.4);
  __m256d a5 = _mm256_set1_pd(1.5);
  __m256d a6 = _mm256_set1_pd(1.6);
  __m256d a7 = _mm256_set1_pd(1.7);
  const __m256d b = _mm256_set1_pd(1.0000001);
  const __m256d c = _mm256_set1_pd(0.9999999);
  for (int64_t i = 0; i < k_peak_iters; ++i) {
    a0 = _mm256_fmadd_pd(a0, b, c);
    a1 = _mm256_fmadd_pd(a1, b, c);
    a2 = _mm256_fmadd_pd(a2, b, c);
    a3 = _mm256_fmadd_pd(a3, b, c);
    a4 = _mm256_fmadd_pd(a4, b, c);
    a5 = _mm256_fmadd_pd(a5, b, c);
    a6 = _mm256_fmadd_pd(a6, b, c);
    a7 = _mm256_fmadd_pd(a7, b, c);
  }
  __m256d sum = _mm256_add_pd(a0, a1);
  sum = _mm256_add_pd(sum, a2);
  sum = _mm256_add_pd(sum, a3);
  sum = _mm256_add_pd(sum, a4);
  sum = _mm256_add_pd(sum, a5);
  sum = _mm256_add_pd(sum, a6);
  sum = _mm256_add_pd(sum, a7);
  alignas(32) double lanes[4];
  _mm256_store_pd(lanes, sum);
  return lanes[0] + lanes[1] + lanes[2] + lanes[3];
}

#else

double peak_fma_block_impl() {
  double a0 = 1.0;
  double a1 = 1.1;
  double a2 = 1.2;
  double a3 = 1.3;
  double a4 = 1.4;
  double a5 = 1.5;
  double a6 = 1.6;
  double a7 = 1.7;
  const double b = 1.0000001;
  const double c = 0.9999999;
  for (int64_t i = 0; i < k_peak_iters; ++i) {
    a0 = a0 * b + c;
    a1 = a1 * b + c;
    a2 = a2 * b + c;
    a3 = a3 * b + c;
    a4 = a4 * b + c;
    a5 = a5 * b + c;
    a6 = a6 * b + c;
    a7 = a7 * b + c;
  }
  return a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

#endif

}  // namespace

int64_t peak_flops_per_call() {
  // 8 independent accumulators, one SIMD width each, mul + add per lane.
#if defined(__AVX512F__)
  constexpr int k_lanes = 8;
#elif defined(__AVX__) && defined(__FMA__)
  constexpr int k_lanes = 4;
#else
  constexpr int k_lanes = 1;
#endif
  return k_peak_iters * 8 * k_lanes * 2;
}

const char* peak_isa_name() {
#if defined(__AVX512F__)
  return "avx512";
#elif defined(__AVX__) && defined(__FMA__)
  return "avx2-fma";
#else
  return "scalar";
#endif
}

double peak_fma_block() { return peak_fma_block_impl(); }

}  // namespace spmv
