#include "mm_io.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>
#include <utility>

namespace spmv {
namespace {

std::string lower_copy(std::string text) {
  for (char& ch : text) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return text;
}

void strip_cr(std::string& line) {
  if (!line.empty() && line.back() == '\r') {
    line.pop_back();
  }
}

std::string trim(const std::string& line) {
  const auto first = line.find_first_not_of(" \t");
  if (first == std::string::npos) {
    return {};
  }
  const auto last = line.find_last_not_of(" \t");
  return line.substr(first, last - first + 1);
}

double parse_double(const std::string& token) {
  std::string text = token;
  for (char& ch : text) {
    if (ch == 'D' || ch == 'd') {
      ch = 'e';
    }
  }
  std::size_t used = 0;
  const double value = std::stod(text, &used);
  if (used != text.size()) {
    throw std::invalid_argument("trailing characters in numeric token");
  }
  return value;
}

class Parser {
 public:
  Parser(std::istream& in, std::string name) : in_(in), name_(std::move(name)) {}

  MtxData parse() {
    const std::string banner = read_banner();
    parse_banner(banner);

    std::string line;
    if (!next_content(line)) {
      fail("missing size line");
    }
    parse_size(line);

    data_.row.reserve(static_cast<std::size_t>(data_.stored_nnz));
    data_.col.reserve(static_cast<std::size_t>(data_.stored_nnz));
    data_.value.reserve(static_cast<std::size_t>(data_.stored_nnz));

    for (int64_t entry = 0; entry < data_.stored_nnz; ++entry) {
      if (!next_content(line)) {
        fail("unexpected end of file while reading entries");
      }
      parse_entry(line);
    }
    if (next_content(line)) {
      fail("extra entries after the declared nnz");
    }

    expand_symmetry();
    return data_;
  }

 private:
  std::istream& in_;
  std::string name_;
  int line_no_ = 0;
  MtxData data_;

  [[noreturn]] void fail(const std::string& message) const {
    throw MtxError(name_, line_no_, message);
  }

  std::string read_banner() {
    std::string line;
    while (std::getline(in_, line)) {
      ++line_no_;
      strip_cr(line);
      const std::string text = trim(line);
      if (text.empty()) {
        continue;
      }
      if (text[0] == '%') {
        if (is_banner(text)) {
          return text;
        }
        continue;
      }
      fail("expected a Matrix Market banner");
    }
    fail("empty file");
  }

  bool next_content(std::string& line) {
    while (std::getline(in_, line)) {
      ++line_no_;
      strip_cr(line);
      line = trim(line);
      if (line.empty() || line[0] == '%') {
        continue;
      }
      return true;
    }
    return false;
  }

  static bool is_banner(const std::string& line) {
    std::istringstream iss(line);
    std::string token;
    if (!(iss >> token)) {
      return false;
    }
    return lower_copy(token).ends_with("matrixmarket");
  }

  void parse_banner(const std::string& line) {
    std::istringstream iss(line);
    std::vector<std::string> tokens;
    std::string token;
    while (iss >> token) {
      tokens.push_back(lower_copy(token));
    }
    if (tokens.size() != 5) {
      fail("banner must have 5 fields: MatrixMarket matrix coordinate <field> <symmetry>");
    }
    if (!tokens[0].ends_with("matrixmarket")) {
      fail("missing MatrixMarket tag");
    }
    if (tokens[1] != "matrix") {
      fail("only matrix objects are supported");
    }
    if (tokens[2] != "coordinate") {
      fail("only coordinate format is supported");
    }
    data_.field = parse_field(tokens[3]);
    data_.symmetry = parse_symmetry(tokens[4]);
  }

  MtxField parse_field(const std::string& token) {
    if (token == "real") {
      return MtxField::Real;
    }
    if (token == "integer") {
      return MtxField::Integer;
    }
    if (token == "pattern") {
      return MtxField::Pattern;
    }
    if (token == "complex") {
      fail("complex matrices are not supported");
    }
    fail("unknown field type '" + token + "'");
  }

  MtxSymmetry parse_symmetry(const std::string& token) {
    if (token == "general") {
      return MtxSymmetry::General;
    }
    if (token == "symmetric") {
      return MtxSymmetry::Symmetric;
    }
    if (token == "skew-symmetric") {
      return MtxSymmetry::SkewSymmetric;
    }
    if (token == "hermitian") {
      return MtxSymmetry::Hermitian;
    }
    fail("unknown symmetry '" + token + "'");
  }

  void parse_size(const std::string& line) {
    std::istringstream iss(line);
    int64_t nrows = 0;
    int64_t ncols = 0;
    int64_t stored = 0;
    if (!(iss >> nrows >> ncols >> stored)) {
      fail("size line must be: nrows ncols nnz");
    }
    std::string extra;
    if (iss >> extra) {
      fail("trailing tokens on the size line");
    }
    if (nrows < 0 || ncols < 0 || stored < 0) {
      fail("negative dimension or nnz");
    }
    const int64_t k_max = std::numeric_limits<int32_t>::max();
    if (nrows > k_max || ncols > k_max) {
      fail("dimension does not fit in int32 column indices");
    }
    data_.nrows = nrows;
    data_.ncols = ncols;
    data_.stored_nnz = stored;
  }

  void parse_entry(const std::string& line) {
    std::istringstream iss(line);
    int64_t row = 0;
    int64_t col = 0;
    if (!(iss >> row >> col)) {
      fail("entry must start with a 1-based row and column");
    }
    double value = 1.0;
    if (data_.field != MtxField::Pattern) {
      std::string token;
      if (!(iss >> token)) {
        fail("entry is missing a value");
      }
      try {
        value = parse_double(token);
      } catch (const std::exception&) {
        fail("could not parse value '" + token + "'");
      }
    }
    std::string extra;
    if (iss >> extra) {
      fail("trailing tokens on an entry");
    }
    if (row < 1 || row > data_.nrows || col < 1 || col > data_.ncols) {
      fail("entry index is outside the declared dimensions");
    }

    const auto row0 = static_cast<int32_t>(row - 1);
    const auto col0 = static_cast<int32_t>(col - 1);
    if (row0 == col0 && data_.symmetry == MtxSymmetry::SkewSymmetric) {
      if (value != 0.0) {
        fail("nonzero skew-symmetric diagonal");
      }
      return;
    }
    data_.row.push_back(row0);
    data_.col.push_back(col0);
    data_.value.push_back(value);
  }

  void expand_symmetry() {
    if (data_.symmetry == MtxSymmetry::General) {
      return;
    }
    const bool skew = data_.symmetry == MtxSymmetry::SkewSymmetric;
    const std::size_t stored = data_.row.size();
    data_.row.reserve(stored * 2);
    data_.col.reserve(stored * 2);
    data_.value.reserve(stored * 2);
    for (std::size_t i = 0; i < stored; ++i) {
      if (data_.row[i] == data_.col[i]) {
        continue;
      }
      const int32_t row = data_.row[i];
      const int32_t col = data_.col[i];
      const double value = data_.value[i];
      data_.row.push_back(col);
      data_.col.push_back(row);
      data_.value.push_back(skew ? -value : value);
    }
  }
};

void validate_csr(const CsrMatrix& matrix) {
  if (matrix.nrows < 0 || matrix.ncols < 0 || matrix.nnz < 0) {
    throw std::runtime_error("CSR has a negative dimension");
  }
  if (matrix.row_ptr.size() != static_cast<std::size_t>(matrix.nrows) + 1) {
    throw std::runtime_error("CSR row_ptr length is not nrows + 1");
  }
  if (matrix.col_idx.size() != static_cast<std::size_t>(matrix.nnz) ||
      matrix.values.size() != static_cast<std::size_t>(matrix.nnz)) {
    throw std::runtime_error("CSR arrays disagree with nnz");
  }
  if (matrix.row_ptr.front() != 0 || matrix.row_ptr.back() != matrix.nnz) {
    throw std::runtime_error("CSR row_ptr does not span nnz entries");
  }
  for (int64_t row = 0; row < matrix.nrows; ++row) {
    const int64_t begin = matrix.row_ptr[static_cast<std::size_t>(row)];
    const int64_t end = matrix.row_ptr[static_cast<std::size_t>(row) + 1];
    if (begin > end) {
      throw std::runtime_error("CSR row_ptr is not nondecreasing");
    }
    for (int64_t k = begin; k < end; ++k) {
      const int32_t col = matrix.col_idx[static_cast<std::size_t>(k)];
      if (col < 0 || col >= matrix.ncols) {
        throw std::runtime_error("CSR column index is out of range");
      }
      if (k > begin && col <= matrix.col_idx[static_cast<std::size_t>(k - 1)]) {
        throw std::runtime_error("CSR row is not strictly sorted");
      }
    }
  }
}

}  // namespace

const char* to_string(MtxSymmetry symmetry) {
  switch (symmetry) {
    case MtxSymmetry::General:
      return "general";
    case MtxSymmetry::Symmetric:
      return "symmetric";
    case MtxSymmetry::SkewSymmetric:
      return "skew-symmetric";
    case MtxSymmetry::Hermitian:
      return "hermitian";
  }
  return "unknown";
}

const char* to_string(MtxField field) {
  switch (field) {
    case MtxField::Real:
      return "real";
    case MtxField::Integer:
      return "integer";
    case MtxField::Pattern:
      return "pattern";
  }
  return "unknown";
}

MtxData read_matrix_market(std::istream& in, const std::string& name) {
  return Parser(in, name).parse();
}

MtxData read_matrix_market(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) {
    throw MtxError(path.string(), 0, "could not open file");
  }
  return read_matrix_market(in, path.string());
}

CsrMatrix to_csr(const MtxData& data) {
  struct Item {
    int32_t row;
    int32_t col;
    double value;
  };

  if (data.row.size() != data.col.size() || data.row.size() != data.value.size()) {
    throw std::runtime_error("triplet arrays have different lengths");
  }

  std::vector<Item> items(data.row.size());
  for (std::size_t i = 0; i < items.size(); ++i) {
    items[i] = Item{data.row[i], data.col[i], data.value[i]};
  }
  std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
    if (a.row != b.row) {
      return a.row < b.row;
    }
    return a.col < b.col;
  });

  std::vector<Item> merged;
  merged.reserve(items.size());
  for (const Item& item : items) {
    if (item.row < 0 || item.row >= data.nrows || item.col < 0 || item.col >= data.ncols) {
      throw std::runtime_error("triplet index out of range while building CSR");
    }
    if (!merged.empty() && merged.back().row == item.row && merged.back().col == item.col) {
      merged.back().value += item.value;
    } else {
      merged.push_back(item);
    }
  }

  CsrMatrix matrix;
  matrix.nrows = data.nrows;
  matrix.ncols = data.ncols;
  matrix.nnz = static_cast<int64_t>(merged.size());
  matrix.row_ptr.assign(static_cast<std::size_t>(matrix.nrows) + 1, 0);
  matrix.col_idx.resize(merged.size());
  matrix.values.resize(merged.size());

  for (const Item& item : merged) {
    matrix.row_ptr[static_cast<std::size_t>(item.row) + 1] += 1;
  }
  for (int64_t row = 0; row < matrix.nrows; ++row) {
    matrix.row_ptr[static_cast<std::size_t>(row) + 1] += matrix.row_ptr[static_cast<std::size_t>(row)];
  }

  std::vector<int64_t> cursor = matrix.row_ptr;
  for (const Item& item : merged) {
    const int64_t pos = cursor[static_cast<std::size_t>(item.row)]++;
    matrix.col_idx[static_cast<std::size_t>(pos)] = item.col;
    matrix.values[static_cast<std::size_t>(pos)] = item.value;
  }

  validate_csr(matrix);
  return matrix;
}

CsrMatrix load_matrix_market(const std::filesystem::path& path) {
  return to_csr(read_matrix_market(path));
}

}  // namespace spmv
