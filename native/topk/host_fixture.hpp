#pragma once

#include "reference.hpp"

namespace strixlab::topk::host_fixture {
// Compiled test faults are unavailable to request JSON, environment or CLI
// flags.
enum class Fault {
  none,
  boundary_tie,
  zero_sign,
  duplicate,
  index,
  count,
  order,
  nan,
  replay_order,
  benchmark_unready
};
struct Gate {
  bool passed = false;
  const char *reason = "not-checked";
  unsigned readiness_checks = 0;
  unsigned setup_calls = 0;
  unsigned operation_calls = 0;
};
Input input();
// Every call establishes fresh local readiness; no incoming receipt or pass
// flag.
Gate check(Fault fault, bool benchmark, bool provider_available);
} // namespace strixlab::topk::host_fixture
