// Separate research fixture. Run only through the reviewed launcher/lease with
// external 30s wall and 20s CPU limits. Not a fixed-v1 benchmark or qualification.
#include "topk_k1.hpp"
#include "baseline/adapter/hip_bitonic_topk.hpp"
#include "reference.hpp"
#include "third_party/nlohmann/json.hpp"
#include <hip/hip_version.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <link.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using nlohmann::json;
namespace cpu = strixlab::topk;
constexpr int launches = 100, samples = 20, warmups = 5;
constexpr std::size_t capacity = 64 * 1024;
constexpr std::size_t allocated_bytes = 2 * capacity * 4 + 2 * 64 * 4;
static_assert(allocated_bytes <= 256 * 1024 * 1024);
void check(hipError_t error, const char* operation) {
    if (error != hipSuccess)
        throw std::runtime_error(std::string(operation) + ": hip_error=" +
                                 std::to_string(static_cast<int>(error)));
}
struct Resources {
    float* src = nullptr;
    int *baseline = nullptr, *candidate = nullptr, *scratch = nullptr;
    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr, stop = nullptr;
    std::vector<std::string> cleanup() {
        std::vector<std::string> errors;
        auto record = [&](hipError_t e, const char* op) {
            if (e != hipSuccess) errors.push_back(std::string(op) + ":" + std::to_string(int(e)));
        };
        if (stream) record(hipStreamSynchronize(stream), "cleanup-stream-sync");
        if (stop) record(hipEventDestroy(stop), "destroy-stop");
        if (start) record(hipEventDestroy(start), "destroy-start");
        if (scratch) record(hipFree(scratch), "free-scratch");
        if (candidate) record(hipFree(candidate), "free-candidate");
        if (baseline) record(hipFree(baseline), "free-baseline");
        if (src) record(hipFree(src), "free-src");
        if (stream) record(hipStreamDestroy(stream), "destroy-stream");
        return errors;
    }
};
std::uint64_t next(std::uint64_t& state) {
    auto z = (state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}
cpu::Input input(int rows, int columns) {
    cpu::Input in{{std::uint64_t(rows), std::uint64_t(columns), 1}, 0,
                  UINT64_C(0x7368727566666c65) ^ (std::uint64_t(rows) << 32) ^ std::uint64_t(columns), {}};
    auto state = in.initial_state;
    std::vector<float> row(columns);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < columns; ++c) row[c] = float(c - columns / 2);
        for (int c = columns - 1; c > 0; --c) std::swap(row[c], row[next(state) % (c + 1)]);
        for (float f : row) {
            std::uint32_t bits;
            std::memcpy(&bits, &f, 4);
            in.bits.push_back(bits);
        }
    }
    if (cpu::preflight(in.shape, in.bits) != cpu::Status::passed ||
        cpu::reference(in.shape, in.bits).status != cpu::Status::passed)
        throw std::runtime_error("CPU preflight/reference failed");
    return in;
}
void launch(Resources& r, bool candidate, int rows, int columns) {
    if (candidate) check(strixlab_topk_k1_hip(r.src, r.candidate, rows, columns, r.stream), "candidate-launch");
    else check(strixlab_baseline_topk_hip(r.src, r.baseline, rows, columns, 1,
                                         r.scratch, capacity, r.stream), "baseline-launch");
}
void batch(Resources& r, bool candidate, int rows, int columns) {
    for (int i = 0; i < launches; ++i) launch(r, candidate, rows, columns);
}
double measure(Resources& r, bool candidate, int rows, int columns) {
    check(hipEventRecord(r.start, r.stream), "record-start");
    batch(r, candidate, rows, columns);
    check(hipEventRecord(r.stop, r.stream), "record-stop");
    check(hipEventSynchronize(r.stop), "sync-stop");
    float ms = 0;
    check(hipEventElapsedTime(&ms, r.start, r.stop), "elapsed-time");
    if (!std::isfinite(ms) || ms <= 0) throw std::runtime_error("invalid-event-duration");
    return double(ms) * 1000 / launches;
}
double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return (values[9] + values[10]) / 2;
}
void validate(Resources& r, const cpu::Input& in, bool candidate) {
    std::vector<int> indices(in.shape.rows, -1);
    check(hipMemcpy(indices.data(), candidate ? r.candidate : r.baseline,
                    indices.size() * sizeof(int), hipMemcpyDeviceToHost), "read-output");
    std::vector<cpu::Pair> pairs;
    for (std::size_t row = 0; row < indices.size(); ++row) {
        int i = indices[row];
        if (i < 0 || std::uint64_t(i) >= in.shape.columns)
            throw std::runtime_error(candidate ? "candidate-invalid-index" : "baseline-invalid-index");
        pairs.push_back({in.bits[row * in.shape.columns + i], std::uint64_t(i)});
    }
    auto status = cpu::validate(in.shape, in.bits, pairs);
    if (status != cpu::Status::passed)
        throw std::runtime_error(std::string(candidate ? "candidate:" : "baseline:") + cpu::status_name(status));
}
struct Loader {
    std::array<std::array<char, 4096>, 256> paths{};
    std::size_t count = 0;
    bool overflow = false;
};
int capture(dl_phdr_info* object, std::size_t, void* context) {
    auto& l = *static_cast<Loader*>(context);
    const char* p = object->dlpi_name ? object->dlpi_name : "";
    const auto size = strnlen(p, 4096);
    if (l.count == l.paths.size() || size == 4096) { l.overflow = true; return 1; }
    std::memcpy(l.paths[l.count++].data(), p, size + 1);
    return 0;
}
void run(Resources& r, json& report) {
    // All six inputs are preflighted before the first device allocation.
    std::vector<cpu::Input> inputs;
    for (int rows : {1, 64}) for (int columns : {32, 256, 1024}) inputs.push_back(input(rows, columns));
    int count = 0, runtime = 0, driver = 0;
    check(hipRuntimeGetVersion(&runtime), "runtime-version");
    check(hipDriverGetVersion(&driver), "driver-version");
    check(hipGetDeviceCount(&count), "device-count");
    report["device_count"] = count;
    if (count != 1) throw std::runtime_error("expected-one-visible-device");
    check(hipSetDevice(0), "set-device");
    hipDeviceProp_t p{};
    check(hipGetDeviceProperties(&p, 0), "device-properties");
    report["device"] = {{"name", std::string(p.name, strnlen(p.name, sizeof(p.name)))},
        {"architecture", std::string(p.gcnArchName, strnlen(p.gcnArchName, sizeof(p.gcnArchName)))},
        {"pci_domain", p.pciDomainID}, {"pci_bus", p.pciBusID}, {"pci_device", p.pciDeviceID}};
    report["hip_runtime_version"] = runtime;
    report["hip_driver_version"] = driver;
    if (std::strncmp(p.gcnArchName, "gfx1151", 7) || (p.gcnArchName[7] && p.gcnArchName[7] != ':'))
        throw std::runtime_error("device-architecture-mismatch");
    check(hipStreamCreate(&r.stream), "create-stream");
    check(hipEventCreate(&r.start), "create-start");
    check(hipEventCreate(&r.stop), "create-stop");
    check(hipMalloc(reinterpret_cast<void**>(&r.src), capacity * sizeof(float)), "allocate-src");
    check(hipMalloc(reinterpret_cast<void**>(&r.baseline), 64 * sizeof(int)), "allocate-baseline");
    check(hipMalloc(reinterpret_cast<void**>(&r.candidate), 64 * sizeof(int)), "allocate-candidate");
    check(hipMalloc(reinterpret_cast<void**>(&r.scratch), capacity * sizeof(int)), "allocate-scratch");
    for (const auto& in : inputs) {
        const int rows = int(in.shape.rows), columns = int(in.shape.columns);
        report["cases"].push_back({{"rows", rows}, {"columns", columns}, {"k", 1},
            {"input_digest", cpu::input_digest(in)}, {"initial_state", in.initial_state},
            {"correctness", false}, {"baseline_us_per_launch", json::array()},
            {"candidate_us_per_launch", json::array()}});
        auto& result = report["cases"].back();
        check(hipMemcpy(r.src, in.bits.data(), in.bits.size() * 4, hipMemcpyHostToDevice), "upload-input");
        check(hipMemset(r.baseline, 0xff, rows * sizeof(int)), "initialize-baseline");
        check(hipMemset(r.candidate, 0xff, rows * sizeof(int)), "initialize-candidate");
        launch(r, false, rows, columns);
        launch(r, true, rows, columns);
        check(hipStreamSynchronize(r.stream), "correctness-sync");
        validate(r, in, false);
        validate(r, in, true);
        result["correctness"] = true;
        for (int i = 0; i < warmups; ++i) {
            batch(r, false, rows, columns);
            batch(r, true, rows, columns);
        }
        check(hipStreamSynchronize(r.stream), "warmup-sync");
        std::vector<double> baseline, candidate;
        for (int i = 0; i < samples; ++i) {
            for (bool is_candidate : {bool(i % 2), !bool(i % 2)}) {
                double us = measure(r, is_candidate, rows, columns);
                (is_candidate ? candidate : baseline).push_back(us);
                result[is_candidate ? "candidate_us_per_launch" : "baseline_us_per_launch"].push_back(us);
            }
        }
        result["baseline_median_us"] = median(baseline);
        result["candidate_median_us"] = median(candidate);
    }
}
} // namespace
int main() {
    Resources resources;
    json report = {{"schema_version", 1}, {"fixture", "rocm10-topk-k1-compare-v1"},
        {"scope", "private-research-only; not fixed-v1 or provider qualification"},
        {"generator", "research-distinct-shuffle-v1"}, {"metadata_coverage", "unknown"},
        {"vendor_authenticity", "unverified"}, {"success", false}, {"comparison_eligible", false},
        {"hip_compile_version", {HIP_VERSION_MAJOR, HIP_VERSION_MINOR, HIP_VERSION_PATCH}},
        {"requested_device_allocation_bytes", allocated_bytes},
        {"external_limits_required", {{"wall_seconds", 30}, {"cpu_seconds", 20}, {"device_bytes", 268435456}}},
        {"warmup_batches_per_provider", warmups}, {"paired_samples", samples}, {"launches_per_batch", launches},
        {"pair_order", "even baseline,candidate; odd candidate,baseline; zero-based"},
        {"timing_scope", "HIP event batch elapsed / 100; all provider stream work including baseline D2D index copy and possible host enqueue gaps; excludes input transfer, validation, allocation, event creation"},
        {"semantics", "K=1 original column index of maximum value, finite F32 distinct values per row; CPU bit-value/index oracle; no tie/NaN/general-K qualification"},
        {"cases", json::array()}, {"failure", nullptr}};
    bool success = false;
    try { run(resources, report); success = true; }
    catch (const std::exception& e) { report["failure"] = e.what(); }
    auto errors = resources.cleanup();
    report["cleanup_errors"] = errors;
    if (!errors.empty()) success = false;
    Loader loader;
    int captured = dl_iterate_phdr(capture, &loader);
    report["loaded_paths"] = json::array();
    for (std::size_t i = 0; i < loader.count; ++i) report["loaded_paths"].push_back(loader.paths[i].data());
    report["loaded_paths_complete"] = captured == 0 && !loader.overflow;
    if (captured != 0 || loader.overflow) success = false;
    report["success"] = success;
    report["comparison_eligible"] = success;
    std::cout << report.dump(-1, ' ', false, json::error_handler_t::replace) << '\n';
    std::cout.flush();
    return success && std::cout.good() ? 0 : 1;
}
