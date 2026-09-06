#include "baseline/adapter/hip_bitonic_topk.hpp"
#include "reference.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <link.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
bool cleanup_failed = false;
hipDeviceProp_t device_properties{};
int runtime_version = 0;
int driver_version = 0;
void check(hipError_t e, const char *where) {
  if (e != hipSuccess)
    throw std::runtime_error(std::string(where) + ": " + hipGetErrorString(e));
}
struct Device {
  void *p = nullptr;
  ~Device() {
    if (p && hipFree(p) != hipSuccess)
      cleanup_failed = true;
  }
};
struct Stream {
  hipStream_t p = nullptr;
  Stream() { check(hipStreamCreate(&p), "hipStreamCreate"); }
  ~Stream() {
    if (p && hipStreamDestroy(p) != hipSuccess)
      cleanup_failed = true;
  }
};
struct Event {
  hipEvent_t p = nullptr;
  Event() { check(hipEventCreate(&p), "hipEventCreate"); }
  ~Event() {
    if (p && hipEventDestroy(p) != hipSuccess)
      cleanup_failed = true;
  }
};
struct Case {
  const char *id;
  strixlab::topk::Shape shape;
};
struct Result {
  const Case *test;
  std::string digest;
  std::vector<float> samples;
};
constexpr size_t max_loaded = 256;
constexpr size_t max_path = 4096;
struct LoadedPaths {
  std::array<std::array<char, max_path>, max_loaded> paths{};
  size_t count = 0;
  bool overflow = false;
};
const Case cases[] = {{"one-row-k1", {1, 1, 1}},
                      {"one-row-small", {1, 8, 3}},
                      {"two-row-small", {2, 8, 8}},
                      {"one-row-padded", {1, 3, 2}},
                      {"one-row-boundary", {1, 1024, 1}}};
strixlab::topk::Input input(const Case &test) {
  strixlab::topk::Input value{test.shape, 0, 0, {}};
  value.bits.resize(test.shape.rows * test.shape.columns);
  for (size_t i = 0; i < value.bits.size(); ++i)
    value.bits[i] = 0x3f800000u + static_cast<uint32_t>(i) * 0x100u;
  return value;
}
Result run_case(const Case &test) {
  using namespace strixlab::topk;
  auto host = input(test);
  if (preflight(test.shape, host.bits) != Status::passed)
    throw std::runtime_error(std::string(test.id) + ": preflight");
  const size_t elements = host.bits.size(),
               output_size = test.shape.rows * test.shape.k;
  int device_count = 0;
  check(hipGetDeviceCount(&device_count), "hipGetDeviceCount");
  if (device_count != 1)
    throw std::runtime_error("expected-one-visible-device");
  check(hipSetDevice(0), "hipSetDevice");
  check(hipGetDeviceProperties(&device_properties, 0),
        "hipGetDeviceProperties");
  if (std::strncmp(device_properties.gcnArchName, "gfx1151", 7) != 0 ||
      (device_properties.gcnArchName[7] != '\0' &&
       device_properties.gcnArchName[7] != ':'))
    throw std::runtime_error("device-architecture-mismatch");
  if (device_properties.pciDomainID != 0 ||
      device_properties.pciBusID != 0xc2 || device_properties.pciDeviceID != 0)
    throw std::runtime_error("device-pci-mismatch");
  Stream stream;
  Device source, output, scratch;
  check(hipMalloc(&source.p, elements * sizeof(uint32_t)), "hipMalloc source");
  check(hipMalloc(&output.p, output_size * sizeof(int)), "hipMalloc output");
  check(hipMalloc(&scratch.p, elements * sizeof(int)), "hipMalloc scratch");
  check(hipMemcpyAsync(source.p, host.bits.data(), elements * sizeof(uint32_t),
                       hipMemcpyHostToDevice, stream.p),
        "copy input");
  auto launch = [&] {
    return strixlab_baseline_topk_hip(
        static_cast<const float *>(source.p), static_cast<int *>(output.p),
        static_cast<int>(test.shape.rows), static_cast<int>(test.shape.columns),
        static_cast<int>(test.shape.k), static_cast<int *>(scratch.p), elements,
        stream.p);
  };
  check(launch(), "correctness launch");
  check(hipStreamSynchronize(stream.p), "correctness sync");
  auto verify_output = [&] {
    std::vector<int> indices(output_size);
    check(hipMemcpy(indices.data(), output.p, output_size * sizeof(int),
                    hipMemcpyDeviceToHost),
          "copy output");
    std::vector<Pair> actual;
    actual.reserve(output_size);
    for (size_t i = 0; i < output_size; ++i) {
      if (indices[i] < 0 ||
          static_cast<size_t>(indices[i]) >= test.shape.columns)
        throw std::runtime_error(std::string(test.id) + ": index bounds");
      actual.push_back(
          {host.bits[(i / test.shape.k) * test.shape.columns + indices[i]],
           static_cast<uint64_t>(indices[i])});
    }
    auto expected = reference(test.shape, host.bits);
    if (expected.status != Status::passed ||
        validate(test.shape, host.bits, actual) != Status::passed ||
        actual.size() != expected.pairs.size())
      throw std::runtime_error(std::string(test.id) + ": oracle mismatch");
    for (size_t i = 0; i < actual.size(); ++i)
      if (actual[i].bits != expected.pairs[i].bits ||
          actual[i].index != expected.pairs[i].index)
        throw std::runtime_error(std::string(test.id) + ": oracle mismatch");
  };
  verify_output();
  Event start, stop;
  check(hipEventRecord(start.p, stream.p), "warmup start");
  check(launch(), "warmup launch");
  check(hipEventRecord(stop.p, stream.p), "warmup stop");
  check(hipEventSynchronize(stop.p), "warmup sync");
  std::vector<float> samples;
  for (int i = 0; i < 5; ++i) {
    check(hipEventRecord(start.p, stream.p), "sample start");
    check(launch(), "sample launch");
    check(hipEventRecord(stop.p, stream.p), "sample stop");
    check(hipEventSynchronize(stop.p), "sample sync");
    float ms = 0;
    check(hipEventElapsedTime(&ms, start.p, stop.p), "sample elapsed");
    if (!std::isfinite(ms) || ms <= 0)
      throw std::runtime_error(std::string(test.id) + ": invalid timing");
    samples.push_back(ms);
  }
  // Validate the measured path too; copies/reference work stay outside event
  // timing.
  verify_output();
  return {&test, input_digest(host), std::move(samples)};
}
int libraries(struct dl_phdr_info *info, size_t, void *data) noexcept {
  auto &loaded = *static_cast<LoadedPaths *>(data);
  const char *name = info->dlpi_name ? info->dlpi_name : "";
  const size_t length = strnlen(name, max_path);
  if (loaded.count == max_loaded || length == max_path) {
    loaded.overflow = true;
    return 1;
  }
  std::memcpy(loaded.paths[loaded.count++].data(), name, length + 1);
  return 0;
}
void json_string(const std::string &value) {
  static constexpr char hex[] = "0123456789abcdef";
  std::cout << '"';
  for (unsigned char byte : value) {
    if (byte == '"' || byte == '\\')
      std::cout << '\\' << byte;
    else if (byte < 0x20 || byte >= 0x7f)
      std::cout << "\\u00" << hex[byte >> 4] << hex[byte & 15];
    else
      std::cout << byte;
  }
  std::cout << '"';
}
} // namespace
int main() {
  static_assert(sizeof(float) == 4 && sizeof(int) == 4);
  try {
    check(hipRuntimeGetVersion(&runtime_version), "hipRuntimeGetVersion");
    check(hipDriverGetVersion(&driver_version), "hipDriverGetVersion");
    std::vector<Result> results;
    for (const auto &test : cases) {
      results.push_back(run_case(test));
      if (cleanup_failed)
        throw std::runtime_error("HIP cleanup failed");
    }
    LoadedPaths libraries_loaded;
    if (dl_iterate_phdr(libraries, &libraries_loaded) != 0 ||
        libraries_loaded.overflow)
      throw std::runtime_error("loaded-library-evidence-limit");
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (const auto &result : results) {
      std::cout << "{\"schema_version\":1,\"success\":true,"
                   "\"scope\":\"finite-distinct-research-only\","
                   "\"metadata_coverage\":\"unknown\",\"vendor_authenticity\":"
                   "\"unverified\","
                   "\"scenario\":\"rocm10-topk-research-v1\",\"id\":\""
                << result.test->id << "\",\"rows\":" << result.test->shape.rows
                << ",\"columns\":" << result.test->shape.columns
                << ",\"k\":" << result.test->shape.k << ",\"input_sha256\":\""
                << result.digest
                << "\",\"device\":\"gfx1151\",\"hip_runtime_version\":"
                << runtime_version
                << ",\"hip_driver_version\":" << driver_version
                << ",\"pci_domain\":0,\"pci_bus\":194,\"pci_device\":0,"
                   "\"warmup_count\":1,\"sample_"
                   "count\":5,\"timing_boundary\":\"adapter-kernel-and-"
                   "hipMemcpy2DAsync\",\"samples_ms\":[";
      for (size_t i = 0; i < result.samples.size(); ++i)
        std::cout << (i ? "," : "") << result.samples[i];
      std::cout << "],\"loaded_paths_encoding\":\"non-ascii-bytes-as-u00xx\","
                   "\"loaded_library_paths\":[";
      for (size_t i = 0; i < libraries_loaded.count; ++i) {
        if (i)
          std::cout << ',';
        json_string(libraries_loaded.paths[i].data());
      }
      std::cout << "]}\n";
    }
    std::cout.flush();
    return std::cout.good() ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "topk-bench-error: " << error.what() << '\n';
    return 1;
  }
}
