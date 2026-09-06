#include "host_fixture.hpp"
#include <stdexcept>
#include <string>
using namespace strixlab::topk::host_fixture;
void require(bool condition) {
  if (!condition)
    throw std::runtime_error("native gate test failed");
}
int main() {
  for (bool benchmark : {false, true}) {
    auto passed = check(Fault::none, benchmark, true);
    require(passed.passed && passed.readiness_checks == 1 &&
            passed.setup_calls == 1 && passed.operation_calls == 2);
    auto unavailable = check(Fault::none, benchmark, false);
    require(!unavailable.passed && unavailable.setup_calls == 0 &&
            unavailable.operation_calls == 0);
    auto nan = check(Fault::nan, benchmark, true);
    require(!nan.passed && nan.readiness_checks == 0 && nan.setup_calls == 0 &&
            nan.operation_calls == 0);
    for (auto fault :
         {Fault::boundary_tie, Fault::zero_sign, Fault::duplicate, Fault::index,
          Fault::count, Fault::order, Fault::replay_order})
      require(!check(fault, benchmark, true).passed);
  }
  require(check(Fault::benchmark_unready, false, true).passed);
  auto invalidated = check(Fault::benchmark_unready, true, true);
  require(!invalidated.passed && invalidated.setup_calls == 0 &&
          invalidated.operation_calls == 0);
}
