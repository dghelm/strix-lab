#include "reference.hpp"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <openssl/evp.h>
#include <sstream>
#include <stdexcept>

namespace strixlab::topk {
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
const std::array<Case, 11> matrix{{
    {"train-r1-c128-k1", "training", {1, 128, 1}},
    {"train-r1-c4096-k20", "training", {1, 4096, 20}},
    {"train-r2-c16381-k8", "training", {2, 16381, 8}},
    {"train-r8-c65536-k20", "training", {8, 65536, 20}},
    {"train-r32-c16384-k64", "training", {32, 16384, 64}},
    {"train-r128-c4096-k256", "training", {128, 4096, 256}},
    {"eval-r1-c262144-k64", "evaluation", {1, 262144, 64}},
    {"eval-r2-c257-k256", "evaluation", {2, 257, 256}},
    {"eval-r8-c16381-k20", "evaluation", {8, 16381, 20}},
    {"eval-r32-c65536-k64", "evaluation", {32, 65536, 64}},
    {"eval-r128-c262144-k20", "evaluation", {128, 262144, 20}},
}};
namespace {
Status shape_status(Shape s) {
  if (!s.rows || !s.columns || !s.k || s.k > s.columns ||
      s.rows > std::numeric_limits<std::size_t>::max() / s.columns)
    return Status::invalid_shape;
  return Status::passed;
}
std::uint64_t next(std::uint64_t &state) {
  auto z = (state += UINT64_C(0x9e3779b97f4a7c15));
  z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
  return z ^ (z >> 31);
}
std::uint32_t encode(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof bits);
  return bits;
}
float decode(std::uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof value);
  return value;
}
} // namespace
Input generate(unsigned ordinal, unsigned family) {
  if (ordinal < 1 || ordinal > matrix.size() || family < 1 || family > 5)
    throw std::invalid_argument("invalid topk case/family");
  Input input{matrix[ordinal - 1].shape,
              family,
              UINT64_C(0x5354524958544f50) ^ (std::uint64_t(ordinal) << 8) ^
                  family,
              {}};
  auto state = input.initial_state;
  constexpr std::uint32_t duplicates[]{0xc0400000, 0xbf800000, 0x80000000,
                                       0,          0x3f800000, 0x40400000};
  constexpr std::uint32_t special[]{0xff800000, 0xbf800000, 0x80000000,
                                    0,          0x3f800000, 0x7f800000,
                                    0xc0000000, 0x40000000};
  input.bits.reserve(input.shape.rows * input.shape.columns);
  for (std::uint64_t i = 0; i < input.shape.rows * input.shape.columns; ++i) {
    auto j = i % input.shape.columns;
    std::uint32_t bits;
    if (family == 1) {
      auto word = next(state);
      bits = ((word >> 63) << 31) | (126u << 23) | (word & 0x7fffff);
    } else if (family == 2)
      bits = duplicates[next(state) % 6];
    else if (family == 3)
      bits = encode(static_cast<float>(j));
    else if (family == 4)
      bits = encode(static_cast<float>(input.shape.columns - j));
    else
      bits = special[i % 8];
    input.bits.push_back(bits);
  }
  return input;
}
Input nan_probe() {
  auto input = generate(1, 1);
  input.bits[0] = 0x7fc00001;
  return input;
}
Status preflight(Shape shape, const std::vector<std::uint32_t> &bits) {
  if (auto status = shape_status(shape); status != Status::passed)
    return status;
  if (bits.size() != shape.rows * shape.columns)
    return Status::invalid_input_size;
  for (auto word : bits)
    if ((word & 0x7f800000) == 0x7f800000 && (word & 0x007fffff))
      return Status::nan_input;
  return Status::passed;
}
std::string input_digest(const Input &input) {
  // NaN probes have identities too: shape/size validation deliberately permits
  // NaN bits.
  if (shape_status(input.shape) != Status::passed ||
      input.bits.size() != input.shape.rows * input.shape.columns)
    throw std::invalid_argument("invalid digest input");
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(),
                                                              EVP_MD_CTX_free);
  if (!ctx || EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) != 1)
    throw std::runtime_error("SHA-256 initialization failed");
  auto update = [&](const void *data, std::size_t size) {
    if (EVP_DigestUpdate(ctx.get(), data, size) != 1)
      throw std::runtime_error("SHA-256 update failed");
  };
  constexpr char domain[] = "strixlab.topk.input.v1";
  update(domain, sizeof domain);
  auto little_endian = [&](std::uint64_t value, unsigned size) {
    unsigned char bytes[8];
    for (unsigned i = 0; i < size; ++i)
      bytes[i] = (value >> (8 * i)) & 255;
    update(bytes, size);
  };
  for (auto value : {input.shape.rows, input.shape.columns, input.shape.k})
    little_endian(value, 8);
  for (auto bits : input.bits)
    little_endian(bits, 4);
  unsigned char digest[EVP_MAX_MD_SIZE];
  unsigned length;
  if (EVP_DigestFinal_ex(ctx.get(), digest, &length) != 1 || length != 32)
    throw std::runtime_error("SHA-256 finalization failed");
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (unsigned i = 0; i < length; ++i)
    out << std::setw(2) << unsigned(digest[i]);
  return out.str();
}
Reference reference(Shape shape, const std::vector<std::uint32_t> &bits) {
  auto status = preflight(shape, bits);
  if (status != Status::passed)
    return {status, {}};
  Reference result{Status::passed, {}};
  result.pairs.reserve(shape.rows * shape.k);
  std::vector<std::uint64_t> indices(shape.columns);
  for (std::uint64_t row = 0; row < shape.rows; ++row) {
    std::iota(indices.begin(), indices.end(), std::uint64_t{0});
    auto offset = row * shape.columns;
    auto before = [&](auto a, auto b) {
      auto av = decode(bits[offset + a]), bv = decode(bits[offset + b]);
      return av > bv || (av == bv && a < b);
    };
    std::partial_sort(indices.begin(), indices.begin() + shape.k, indices.end(),
                      before);
    for (std::uint64_t i = 0; i < shape.k; ++i)
      result.pairs.push_back({bits[offset + indices[i]], indices[i]});
  }
  return result;
}
Status validate(Shape shape, const std::vector<std::uint32_t> &bits,
                const std::vector<Pair> &output) {
  auto status = preflight(shape, bits);
  if (status != Status::passed)
    return status;
  if (output.size() != shape.rows * shape.k)
    return Status::invalid_output_size;
  for (std::uint64_t row = 0; row < shape.rows; ++row) {
    std::vector<bool> seen(shape.columns, false);
    for (std::uint64_t i = 0; i < shape.k; ++i) {
      auto pair = output[row * shape.k + i];
      if (pair.index >= shape.columns)
        return Status::invalid_index;
      if (seen[pair.index])
        return Status::duplicate_index;
      seen[pair.index] = true;
      if (pair.bits != bits[row * shape.columns + pair.index])
        return Status::value_mismatch;
    }
  }
  auto expected = reference(shape, bits);
  for (std::size_t i = 0; i < output.size(); ++i)
    if (output[i].index != expected.pairs[i].index)
      return Status::reference_mismatch;
  return Status::passed;
}
const char *status_name(Status status) {
  switch (status) {
  case Status::passed:
    return "passed";
  case Status::invalid_shape:
    return "invalid-shape";
  case Status::invalid_input_size:
    return "invalid-input-size";
  case Status::nan_input:
    return "nan-input";
  case Status::invalid_output_size:
    return "invalid-output-size";
  case Status::invalid_index:
    return "invalid-index";
  case Status::duplicate_index:
    return "duplicate-index";
  case Status::value_mismatch:
    return "value-mismatch";
  case Status::reference_mismatch:
    return "reference-mismatch";
  }
  throw std::invalid_argument("unknown status");
}
} // namespace strixlab::topk
