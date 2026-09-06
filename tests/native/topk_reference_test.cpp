#include "reference.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace strixlab::topk;
void require(bool condition) {
  if (!condition)
    throw std::runtime_error("native reference check failed");
}
void check(Shape shape, const std::vector<std::uint32_t> &bits,
           const std::vector<std::uint64_t> &indices) {
  auto result = reference(shape, bits);
  require(result.status == Status::passed &&
          result.pairs.size() == indices.size());
  for (std::size_t i = 0; i < indices.size(); ++i) {
    require(result.pairs[i].index == indices[i]);
    require(result.pairs[i].bits ==
            bits[(i / shape.k) * shape.columns + indices[i]]);
  }
  require(validate(shape, bits, result.pairs) == Status::passed);
}
void self_test() {
  // Boundary membership: the second row also proves indices are row-local.
  check({2, 5, 2},
        {0x3f800000, 0x40400000, 0x40400000, 0x40400000, 0, 0x40400000, 0,
         0x40400000, 0x40400000, 0},
        {1, 2, 0, 2});
  check({1, 5, 3}, {0x80000000, 0, 0x80000000, 0, 0xbf800000}, {0, 1, 2});
  check({1, 6, 6},
        {0xff800000, 0, 0x7f800000, 0x80000000, 0x7f800000, 0xff800000},
        {2, 4, 1, 3, 0, 5});
  check({1, 4, 2}, {0xbf800000, 0xc0400000, 0xc0000000, 0xbf800000}, {0, 3});
  // All binary32 NaN classes/signs, including a late row, must fail preflight.
  for (auto nan : {0x7fc00001u, 0xffc00001u, 0x7f800001u, 0xff800001u}) {
    std::vector<std::uint32_t> bits{0, 0, 0, nan};
    require(preflight({2, 2, 1}, bits) == Status::nan_input);
    auto result = reference({2, 2, 1}, bits);
    require(result.status == Status::nan_input && result.pairs.empty());
    require(validate({2, 2, 1}, bits, {}) == Status::nan_input);
  }
  require(std::string(status_name(Status::nan_input)) == "nan-input");
  auto probe = nan_probe();
  require(probe.initial_state == UINT64_C(0x5354524958544e51));
  require(probe.family == 1 && probe.bits[0] == 0x7fc00001);
  require(preflight(probe.shape, probe.bits) == Status::nan_input);
  require(input_digest(probe) ==
          "bb2cd21188a4c2d25003df1cd66c9d8adf8be217144f3f1027efd2d5e7b7be4c");
  for (auto shape :
       {Shape{0, 2, 1}, Shape{1, 0, 1}, Shape{1, 2, 0}, Shape{1, 2, 3},
        Shape{std::numeric_limits<std::uint64_t>::max(), 2, 1}})
    require(preflight(shape, {}) == Status::invalid_shape);
  require(preflight({1, 2, 1}, {0}) == Status::invalid_input_size);
  Shape shape{1, 4, 2};
  std::vector<std::uint32_t> bits{0, 0x80000000, 0, 0};
  require(validate(shape, bits, {}) == Status::invalid_output_size);
  require(validate(shape, bits, {{0, 0}, {0, 4}}) == Status::invalid_index);
  require(validate(shape, bits, {{0, 0}, {0, 0}}) == Status::duplicate_index);
  require(validate(shape, bits, {{0, 0}, {0, 1}}) == Status::value_mismatch);
  require(validate(shape, bits, {{0, 0}, {0, 2}}) ==
          Status::reference_mismatch);
  require(validate(shape, bits, {{0x80000000, 1}, {0, 0}}) ==
          Status::reference_mismatch);
  require(validate(shape, bits, {{0, 0}, {0x80000000, 1}, {0, 2}}) ==
          Status::invalid_output_size);
  for (auto args :
       {std::pair<unsigned, unsigned>{0, 1}, {12, 1}, {1, 0}, {1, 6}}) {
    bool rejected = false;
    try {
      generate(args.first, args.second);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    require(rejected);
  }
  auto malformed = generate(1, 1);
  malformed.bits.pop_back();
  bool rejected = false;
  try {
    input_digest(malformed);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected);
}
int main(int argc, char **argv) {
  try {
    self_test();
    if (argc == 1)
      return 0;
    if (argc == 2 && std::string(argv[1]) == "matrix") {
      for (const auto &c : matrix)
        std::cout << c.id << ' ' << c.set << ' ' << c.shape.rows << ' '
                  << c.shape.columns << ' ' << c.shape.k << '\n';
      return 0;
    }
    if (argc == 3) {
      auto input = generate(std::stoul(argv[1]), std::stoul(argv[2]));
      auto result = reference(input.shape, input.bits);
      require(result.status == Status::passed);
      std::cout << input.initial_state << ' ' << input_digest(input) << '\n';
      for (auto bits : input.bits)
        std::cout << bits << ' ';
      std::cout << '\n';
      for (auto p : result.pairs)
        std::cout << p.index << ':' << p.bits << ' ';
      std::cout << '\n';
      return 0;
    }
    throw std::invalid_argument("expected [matrix | ordinal family]");
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
