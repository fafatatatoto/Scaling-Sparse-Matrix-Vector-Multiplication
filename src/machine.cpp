#include "machine.hpp"

#include "peak.hpp"
#include "stream.hpp"
#include "timed_loop.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

namespace spmv {
namespace {

volatile double g_observed = 0;

std::string trim(std::string text) {
  const auto first = text.find_first_not_of(" \t");
  if (first == std::string::npos) {
    return {};
  }
  const auto last = text.find_last_not_of(" \t\r");
  return text.substr(first, last - first + 1);
}

std::string cpu_model() {
  std::ifstream in("/proc/cpuinfo");
  std::string line;
  while (std::getline(in, line)) {
    constexpr const char* k_key = "model name";
    if (line.compare(0, 10, k_key) != 0) {
      continue;
    }
    const auto pos = line.find(':');
    if (pos == std::string::npos) {
      continue;
    }
    return trim(line.substr(pos + 1));
  }
  return "unknown";
}

int64_t parse_cache_size(const std::string& text) {
  std::size_t used = 0;
  const long long value = std::stoll(text, &used);
  while (used < text.size() && std::isspace(static_cast<unsigned char>(text[used]))) {
    ++used;
  }
  int64_t scale = 1;
  if (used < text.size()) {
    const char unit = text[used];
    if (unit == 'K' || unit == 'k') {
      scale = 1024;
    } else if (unit == 'M' || unit == 'm') {
      scale = 1024LL * 1024;
    } else if (unit == 'G' || unit == 'g') {
      scale = 1024LL * 1024 * 1024;
    }
  }
  return static_cast<int64_t>(value) * scale;
}

std::optional<int64_t> read_cache_index(int index) {
  const std::string path =
      "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(index) + "/size";
  std::ifstream in(path);
  std::string text;
  if (!(in >> text)) {
    return std::nullopt;
  }
  return parse_cache_size(text);
}

int64_t llc_bytes() {
  if (const auto llc = read_cache_index(3)) {
    return *llc;
  }
  if (const auto l2 = read_cache_index(2)) {
    return *l2;
  }
  return 8LL * 1024 * 1024;
}

int64_t stream_length(int64_t cache_bytes) {
  int64_t per_array = std::max(cache_bytes * 4, int64_t{32} << 20);
  constexpr int64_t k_cap = int64_t{256} << 20;
  if (per_array > k_cap) {
    per_array = k_cap;
  }
  return per_array / static_cast<int64_t>(sizeof(double));
}

}  // namespace

MachineRoof measure_machine(const BenchConfig& config) {
  MachineRoof roof;
  roof.cpu = cpu_model();
  roof.llc_bytes = llc_bytes();
  roof.peak_isa = peak_isa_name();
  roof.stream_n = stream_length(roof.llc_bytes);

  const double flops = static_cast<double>(peak_flops_per_call());
  const timed::Samples peak = timed::run(config, [] { g_observed = peak_fma_block(); });
  roof.peak_median_s = peak.median_s;
  if (peak.median_s > 0.0) {
    roof.peak_gflops = flops / peak.median_s / 1e9;
  }

  std::vector<double> a(static_cast<std::size_t>(roof.stream_n), 0.0);
  std::vector<double> b(static_cast<std::size_t>(roof.stream_n), 1.0);
  std::vector<double> c(static_cast<std::size_t>(roof.stream_n), 1.1);
  const double scalar = 1.0001;
  const timed::Samples stream = timed::run(config, [&] {
    stream_triad(a.data(), b.data(), c.data(), roof.stream_n, scalar);
    g_observed = a[0];
  });
  roof.stream_median_s = stream.median_s;
  roof.stream_bytes = static_cast<double>(k_stream_bytes_per_element) * static_cast<double>(roof.stream_n);
  if (stream.median_s > 0.0) {
    roof.stream_gbps = roof.stream_bytes / stream.median_s / 1e9;
  }
  return roof;
}

}  // namespace spmv
