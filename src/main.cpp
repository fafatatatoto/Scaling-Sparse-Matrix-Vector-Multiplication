#include "benchmark.hpp"
#include "check.hpp"
#include "dense.hpp"
#include "machine.hpp"
#include "mm_io.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

constexpr double k_check_rel_tol = 1e-6;

void usage(std::ostream& out) {
  out << R"(Usage: spmv <matrix.mtx> [options]
       spmv --dense N [options]
       spmv --machine [options]

Serial CSR SpMV benchmark. Times y = A*x with x = 1.
--dense N times a dense n-by-n matvec filled with ones.
--machine measures the single-thread FMA peak and STREAM triad roof.

Options:
  --check              compare CSR against a triplet reference, then benchmark
  --csv                print a CSV header and one row on stdout
  --csv-out FILE       append one CSV row to FILE
  --warmup N           untimed batches before and after calibration (default 2)
  --min-samples N      minimum timed samples (default 7)
  --max-samples N      maximum timed samples (default 100)
  --min-seconds S      keep sampling until this much timed work (default 1)
  --sample-seconds S   target length of one timed sample (default 0.01)
)";
}

std::string require_value(int& index, int argc, char** argv, const char* option) {
  if (index + 1 >= argc) {
    throw std::runtime_error(std::string(option) + " needs a value");
  }
  ++index;
  return argv[index];
}

std::string csv_quote(const std::string& text) {
  std::string out;
  out.reserve(text.size() + 2);
  out.push_back('"');
  for (char ch : text) {
    if (ch == '"') {
      out.push_back('"');
    }
    out.push_back(ch);
  }
  out.push_back('"');
  return out;
}

// Column order is part of the result format. Keep it in sync with docs/methodology.md.
const char* csv_header() {
  return "matrix,nrows,ncols,stored_nnz,expanded_nnz,nnz,symmetry,field,"
         "median_s,min_s,max_s,gflops,"
         "bytes_x_reused,bytes_x_reloaded,gbps_x_reused,gbps_x_reloaded,"
         "ai_x_reused,ai_x_reloaded,unique_cols,samples,batch";
}

std::string csv_row_raw(const std::string& matrix, long long nrows, long long ncols, long long stored,
                        long long expanded, long long nnz, const char* symmetry, const char* field,
                        const spmv::BenchResult& result) {
  std::ostringstream out;
  out << std::setprecision(16) << std::scientific;
  out << csv_quote(matrix) << ',' << nrows << ',' << ncols << ',' << stored << ',' << expanded << ','
      << nnz << ',' << symmetry << ',' << field << ',' << result.median_s << ',' << result.min_s << ','
      << result.max_s << ',' << result.gflops << ',' << result.bytes_x_reused << ','
      << result.bytes_x_reloaded << ',' << result.gbps_x_reused << ',' << result.gbps_x_reloaded << ','
      << result.ai_x_reused << ',' << result.ai_x_reloaded << ',' << result.unique_cols << ','
      << result.samples << ',' << result.batch;
  return out.str();
}

std::string csv_row(const std::string& matrix, const spmv::MtxData& mtx, const spmv::CsrMatrix& A,
                    const spmv::BenchResult& result) {
  return csv_row_raw(matrix, A.nrows, A.ncols, mtx.stored_nnz, static_cast<long long>(mtx.row.size()),
                     A.nnz, spmv::to_string(mtx.symmetry), spmv::to_string(mtx.field), result);
}

void print_int(const char* key, long long value) {
  std::cout << key << ": " << value << '\n';
}

void print_float(const char* key, double value) {
  std::cout << std::scientific << std::setprecision(6) << key << ": " << value << '\n';
}

void print_text(const char* key, const std::string& value) {
  std::cout << key << ": " << value << '\n';
}

void write_csv_file(const std::filesystem::path& path, const std::string& header, const std::string& row) {
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }
  const bool write_header = !std::filesystem::exists(path) || std::filesystem::file_size(path) == 0;
  std::ofstream out(path, std::ios::app);
  if (!out) {
    throw std::runtime_error("could not open " + path.string());
  }
  if (write_header) {
    out << header << '\n';
  }
  out << row << '\n';
}

void print_benchmark(const spmv::BenchResult& result) {
  print_int("unique_cols", result.unique_cols);
  print_float("median_s", result.median_s);
  print_float("min_s", result.min_s);
  print_float("max_s", result.max_s);
  print_float("gflops", result.gflops);
  print_float("ai_x_reused", result.ai_x_reused);
  print_float("ai_x_reloaded", result.ai_x_reloaded);
  print_float("gbps_x_reused", result.gbps_x_reused);
  print_float("gbps_x_reloaded", result.gbps_x_reloaded);
  print_float("bytes_x_reused", result.bytes_x_reused);
  print_float("bytes_x_reloaded", result.bytes_x_reloaded);
  print_int("samples", result.samples);
  print_int("batch", result.batch);
  print_text("compiler", spmv::compiler_id());
  print_text("build", spmv::build_flags());
}

const char* machine_header() {
  return "cpu,llc_bytes,peak_isa,peak_gflops,stream_gbps,stream_n,stream_bytes,peak_median_s,stream_median_s";
}

std::string machine_row(const spmv::MachineRoof& roof) {
  std::ostringstream out;
  out << std::setprecision(16) << std::scientific;
  out << csv_quote(roof.cpu) << ',' << roof.llc_bytes << ',' << roof.peak_isa << ',' << roof.peak_gflops
      << ',' << roof.stream_gbps << ',' << roof.stream_n << ',' << roof.stream_bytes << ','
      << roof.peak_median_s << ',' << roof.stream_median_s;
  return out.str();
}

int run_machine(bool csv, const std::string& csv_out, const spmv::BenchConfig& config) {
  const spmv::MachineRoof roof = spmv::measure_machine(config);
  const std::string row = machine_row(roof);
  if (csv) {
    std::cout << machine_header() << '\n' << row << '\n';
  } else {
    std::cout << "# single-thread roofs; see docs/methodology.md\n";
    print_text("cpu", roof.cpu);
    print_int("llc_bytes", roof.llc_bytes);
    print_text("peak_isa", roof.peak_isa);
    print_float("peak_gflops", roof.peak_gflops);
    print_float("stream_gbps", roof.stream_gbps);
    print_int("stream_n", roof.stream_n);
    print_float("stream_bytes", roof.stream_bytes);
    print_float("peak_median_s", roof.peak_median_s);
    print_float("stream_median_s", roof.stream_median_s);
    print_text("compiler", spmv::compiler_id());
    print_text("build", spmv::build_flags());
  }
  if (!csv_out.empty()) {
    write_csv_file(csv_out, machine_header(), row);
  }
  return 0;
}

int run_dense(int64_t n, bool csv, const std::string& csv_out, const spmv::BenchConfig& config) {
  const spmv::BenchResult result = spmv::run_dense_benchmark(n, config);
  const long long nn = n * n;
  const std::string row = csv_row_raw("dense", n, n, nn, nn, nn, "dense", "real", result);
  if (csv) {
    std::cout << csv_header() << '\n' << row << '\n';
  } else {
    std::cout << "# dense matvec; both x models count one sequential read of x\n";
    print_text("matrix", "dense");
    print_text("symmetry", "dense");
    print_text("field", "real");
    print_int("nrows", n);
    print_int("ncols", n);
    print_int("nnz", nn);
    print_benchmark(result);
  }
  if (!csv_out.empty()) {
    write_csv_file(csv_out, csv_header(), row);
  }
  return 0;
}

int run(int argc, char** argv) {
  std::string matrix_path;
  bool check = false;
  bool csv = false;
  bool machine = false;
  int dense_n = -1;
  std::string csv_out;
  spmv::BenchConfig config;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      usage(std::cout);
      return 0;
    }
    if (arg == "--check") {
      check = true;
    } else if (arg == "--csv") {
      csv = true;
    } else if (arg == "--machine") {
      machine = true;
    } else if (arg == "--dense") {
      dense_n = std::stoi(require_value(i, argc, argv, "--dense"));
    } else if (arg == "--csv-out") {
      csv_out = require_value(i, argc, argv, "--csv-out");
    } else if (arg == "--warmup") {
      config.warmup_batches = std::stoi(require_value(i, argc, argv, "--warmup"));
    } else if (arg == "--min-samples") {
      config.min_samples = std::stoi(require_value(i, argc, argv, "--min-samples"));
    } else if (arg == "--max-samples") {
      config.max_samples = std::stoi(require_value(i, argc, argv, "--max-samples"));
    } else if (arg == "--min-seconds") {
      config.min_seconds = std::stod(require_value(i, argc, argv, "--min-seconds"));
    } else if (arg == "--sample-seconds") {
      config.target_sample_seconds = std::stod(require_value(i, argc, argv, "--sample-seconds"));
    } else if (!arg.empty() && arg[0] == '-') {
      throw std::runtime_error("unknown option " + arg);
    } else if (matrix_path.empty()) {
      matrix_path = arg;
    } else {
      throw std::runtime_error("unexpected argument " + arg);
    }
  }

  const int modes = static_cast<int>(machine) + static_cast<int>(dense_n >= 0) + static_cast<int>(!matrix_path.empty());
  if (modes != 1) {
    usage(std::cerr);
    return 1;
  }
  if (machine) {
    return run_machine(csv, csv_out, config);
  }
  if (dense_n >= 0) {
    return run_dense(dense_n, csv, csv_out, config);
  }

  const spmv::MtxData mtx = spmv::read_matrix_market(matrix_path);
  const spmv::CsrMatrix matrix = spmv::to_csr(mtx);

  if (check) {
    const spmv::SpmvCheck compared = spmv::check_csr_against_mtx(matrix, mtx);
    if (compared.max_rel_err > k_check_rel_tol) {
      std::cerr << std::scientific << std::setprecision(6)
                << "check failed: max_rel_err=" << compared.max_rel_err
                << " max_abs_err=" << compared.max_abs_err << '\n';
      return 2;
    }
    if (!csv) {
      print_float("check_max_rel_err", compared.max_rel_err);
      print_float("check_max_abs_err", compared.max_abs_err);
    }
  }

  const spmv::BenchResult result = spmv::run_benchmark(matrix, config);
  const std::string row = csv_row(matrix_path, mtx, matrix, result);

  if (csv) {
    std::cout << csv_header() << '\n' << row << '\n';
  } else {
    std::cout << "# model gbps and ai use docs/methodology.md, not hardware counters\n";
    print_text("matrix", matrix_path);
    print_text("symmetry", spmv::to_string(mtx.symmetry));
    print_text("field", spmv::to_string(mtx.field));
    print_int("nrows", matrix.nrows);
    print_int("ncols", matrix.ncols);
    print_int("stored_nnz", mtx.stored_nnz);
    print_int("expanded_nnz", static_cast<long long>(mtx.row.size()));
    print_int("nnz", matrix.nnz);
    print_benchmark(result);
  }

  if (!csv_out.empty()) {
    write_csv_file(csv_out, csv_header(), row);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << '\n';
    return 1;
  }
}
