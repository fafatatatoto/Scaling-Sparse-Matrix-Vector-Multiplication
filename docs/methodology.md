# Benchmark methodology

This file is the contract for later figures. If a definition changes, change it here and rerun. Do not silently compare numbers from two definitions.

## Kernel

Serial CSR, double precision, `y = Ax`. Each stored nonzero after symmetric expansion contributes one multiply and one add.

Timed runs use `x = 1`. That vector is reproducible and avoids denormals. It does not check column indices: with every `x[j]` equal, a wrong column still produces the row sum. `--check` uses `x[j] = 1 + 0.25 * (j % 17)` and a separate triplet accumulator.

## Compiler

Release build:

```text
-O3 -march=native
```

C++20. No `-ffast-math`. No link-time optimization. LTO can see that every call writes the same `y` and delete the repeated calls. The timed loop and the kernel are in separate translation units for the same reason.

Debug builds are for tests. Performance numbers come from Release.

## Time

Clock is `std::chrono::steady_clock`.

One untimed SpMV runs first so the vectors are faulted in. The harness then chooses a batch size so one sample lasts about 10 ms. The batch caps at `2^20`. It records per-SpMV times until it has at least 7 samples and at least 1 second of timed work, whichever needs more iterations, and it stops at 100 samples. The reported time is the median. `min_s` and `max_s` are there so a noisy run is visible.

`benchmarks/tiny.mtx` checks the pipeline. Its runtime is call overhead. Performance claims use matrices whose working set is large enough to miss in cache.

## Rates

```text
gflops = 2 * nnz / median_s / 1e9
```

`1e9` is a decimal giga, not `2^30`.

Model bytes, per SpMV, from the storage the program actually allocates:

```text
values          8 * nnz
col_idx         4 * nnz          int32
row_ptr         8 * (nrows + 1)  int64
y               16 * nrows       write-allocate: a read and a write
x reused        8 * unique_cols  each touched x element loaded once
x reloaded      8 * nnz          every nonzero loads x again
```

```text
bytes_x_reused    = values + col_idx + row_ptr + y + x reused
bytes_x_reloaded  = values + col_idx + row_ptr + y + x reloaded
gbps_*            = bytes_* / median_s / 1e9
ai_*              = 2 * nnz / bytes_*        FLOP/byte
```

`ai_x_reused` is the higher intensity (optimistic locality). `ai_x_reloaded` is the lower intensity (no reuse of `x`). Measured traffic for an irregular matrix sits between them.

These GB/s numbers are model estimates. They are not `perf` or LIKWID. Hardware counters belong to the locality milestone.

`row_ptr` is 8 bytes so a matrix with more than `2^31 - 1` nonzeros can still be indexed. Column indices are 4 bytes, so `nrows` and `ncols` must be at most `2^31 - 1`.

The code for this model is `src/traffic.cpp`. A test locks the byte count of `benchmarks/tiny.mtx` at 200 (x reused) and 224 (x reloaded).

## Matrix Market

Supported: coordinate format, field `real`, `integer`, or `pattern`, symmetry `general`, `symmetric`, `skew-symmetric`, or real `hermitian`.

Symmetric and Hermitian files store one triangle. The loader mirrors off-diagonal entries. `stored_nnz` is the header count. `expanded_nnz` is the triplet count after mirroring. `nnz` is the CSR length after duplicate `(row, col)` entries are summed. FLOP/s and traffic use `nnz`, because that is the work the kernel does.

SuiteSparse reports the stored triangle for symmetric matrices. `stored_nnz` is the number that matches the website. `nnz` is larger by the mirrored off-diagonals.

Explicit zeros are kept. They occupy memory and count as nonzeros. A zero skew-symmetric diagonal is structural and is dropped. A nonzero skew-symmetric diagonal is an error.

Duplicate coordinates are summed in sorted `(row, col)` order, not file order.

Fortran `D` exponents (`2.5D+01`) are accepted.

## CSV

`--csv` writes this header. Plot scripts select columns by name.

```text
matrix,nrows,ncols,stored_nnz,expanded_nnz,nnz,symmetry,field,median_s,min_s,max_s,gflops,bytes_x_reused,bytes_x_reloaded,gbps_x_reused,gbps_x_reloaded,ai_x_reused,ai_x_reloaded,unique_cols,samples,batch
```

## Correctness tolerance

`--check` fails when the maximum relative error against the triplet reference exceeds `1e-6`. The scale is `max(1, |y_ref|)`, so tiny rows are judged on an absolute tolerance of `1e-6`.

## Dense matvec

`--dense N` times `y = Ax` for an N-by-N row-major matrix filled with ones, and the same `x = 1`. Each entry is one multiply and one add, so flops are `2 N^2`. The all-ones product is the vector of N's and is checked before timing.

Traffic for this kernel, in bytes:

```text
A    8 * N^2     read once, streamed
x    8 * N       read once; for N = 4096 this vector is 32 KiB and stays in cache
y    16 * N      write-allocate
```

The two CSR columns `bytes_x_reused` and `bytes_x_reloaded` are both set to that sum. Dense has one traffic model because `x` is scanned sequentially. Arithmetic intensity approaches 0.25 FLOP/byte as N grows. N is capped at 16384 so the matrix stays within 2 GiB.

## Single-thread roofs

`--machine` measures the roofs used on the serial roofline. Both kernels are single-threaded, matching serial SpMV.

- Compute roof: eight independent register FMAs, selected from AVX-512, AVX2 FMA, or scalar by the compiler's `-march=native` macros. An FMA counts as two flops. This is a measured rate, not a vendor peak.
- Bandwidth roof: a triad `a[i] = b[i] + scalar * c[i]`. Each array is at least four times the last-level cache and at most 256 MiB. Bytes follow the STREAM convention, `24 * n` (two reads and one write), not the write-allocate count used for SpMV's `y`.

The roofline y-axis is measured GFLOP/s. The x-axis for CSR is the segment from `ai_x_reloaded` to `ai_x_reused`. A point above the STREAM line means the compulsory bytes were served from cache faster than DRAM, which is expected when the working set fits in the last-level cache.

Machine CSV header:

```text
cpu,llc_bytes,peak_isa,peak_gflops,stream_gbps,stream_n,stream_bytes,peak_median_s,stream_median_s
```
