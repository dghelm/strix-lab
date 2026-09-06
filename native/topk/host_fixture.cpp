#include "host_fixture.hpp"

#include <algorithm>

namespace strixlab::topk::host_fixture {
Input input() {
  // Independent of topk-input-v1: tiny transport-only data with a tie at K.
  return {
      {1, 6, 3}, 0, 0, {0xff800000, 0x7f800000, 0x80000000, 0, 0, 0x7f800000}};
}
Gate check(Fault fault, bool benchmark, bool provider_available) {
  Gate gate;
  auto data = input();
  if (fault == Fault::nan)
    data.bits.back() = 0x7f800001;
  auto status = preflight(data.shape, data.bits);
  if (status != Status::passed) {
    gate.reason = status_name(status);
    return gate;
  }
  ++gate.readiness_checks;
  // No device or graph exists here. Only this explicit CPU fixture can be
  // ready.
  if (!provider_available) {
    gate.reason = "provider-unavailable";
    return gate;
  }
  if (benchmark && fault == Fault::benchmark_unready) {
    gate.reason = "fixture-readiness-failed";
    return gate;
  }
  ++gate.setup_calls;
  std::vector<Pair> direct;
  for (unsigned mode = 0; mode < 2; ++mode) {
    ++gate.operation_calls;
    // Hand-specified output, never copied from the reference being tested.
    std::vector<Pair> output{{0x7f800000, 1}, {0x7f800000, 5}, {0x80000000, 2}};
    switch (fault) {
    case Fault::boundary_tie:
      output.back() = {0, 3};
      break;
    case Fault::zero_sign:
      output.back().bits = 0;
      break;
    case Fault::duplicate:
      output[1] = output[0];
      break;
    case Fault::index:
      output.back().index = 6;
      break;
    case Fault::count:
      output.pop_back();
      break;
    case Fault::order:
      std::swap(output[0], output[1]);
      break;
    case Fault::replay_order:
      if (mode == 1)
        std::swap(output[0], output[1]);
      break;
    default:
      break;
    }
    status = validate(data.shape, data.bits, output);
    if (status != Status::passed) {
      gate.reason = status_name(status);
      return gate;
    }
    if (mode == 0)
      direct = output;
    else if (!std::equal(direct.begin(), direct.end(), output.begin(),
                         output.end(), [](Pair a, Pair b) {
                           return a.index == b.index && a.bits == b.bits;
                         })) {
      gate.reason = "fixture-replay-mismatch";
      return gate;
    }
  }
  gate.passed = true;
  gate.reason = "passed";
  return gate;
}
} // namespace strixlab::topk::host_fixture
