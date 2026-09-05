#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace strixlab::topk {
inline constexpr char generator_id[] = "topk-input-v1";
struct Shape {
  std::uint64_t rows, columns, k;
};
struct Case {
  const char *id;
  const char *set;
  Shape shape;
};
extern const std::array<Case, 11> matrix;
struct Input {
  Shape shape;
  unsigned family;
  std::uint64_t initial_state;
  std::vector<std::uint32_t> bits;
};
struct Pair {
  std::uint32_t bits;
  std::uint64_t index;
};
enum class Status {
  passed,
  invalid_shape,
  invalid_input_size,
  nan_input,
  invalid_output_size,
  invalid_index,
  duplicate_index,
  value_mismatch,
  reference_mismatch
};
const char *status_name(Status status);
Input generate(unsigned case_ordinal, unsigned family);
Input nan_probe();
std::string input_digest(const Input &input);
// Preflight is allocation-free and must precede future device
// allocation/capture/launch.
Status preflight(Shape shape, const std::vector<std::uint32_t> &bits);
struct Reference {
  Status status;
  std::vector<Pair> pairs;
};
Reference reference(Shape shape, const std::vector<std::uint32_t> &bits);
Status validate(Shape shape, const std::vector<std::uint32_t> &bits,
                const std::vector<Pair> &output);
} // namespace strixlab::topk
