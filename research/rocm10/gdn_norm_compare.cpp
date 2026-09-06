// Source-derived c7af5c6 corrected GDN normalization experiment, not full-backend qualification.
// Use reviewed launcher only: 30s wall, 20s CPU, 256MiB allocation ceiling.
#include "gdn_norm.hpp"
#include "third_party/nlohmann/json.hpp"
#include <hip/hip_version.h>
#include <openssl/sha.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <link.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using nlohmann::json;
namespace gdn = strixlab_gdn_norm;
constexpr int launches = 100, samples = 20, warmups = 5;
constexpr int width = 128, heads = 16, token_stride = 8192, k_offset = 2048;
constexpr std::size_t source_capacity = 512 * token_stride;
constexpr std::size_t output_capacity = 512 * heads * width;
constexpr std::size_t guard = 32;
constexpr std::size_t allocated_bytes = (source_capacity + 6 * output_capacity + 5 * guard) * sizeof(float);
static_assert(allocated_bytes < 256 * 1024 * 1024);
constexpr float epsilon = 1.0e-6f;
// FP32 unit roundoff u=2^-24: even a conservative 128-term summation has
// gamma_128 < 7.7e-6. Positive squares, division, rsqrt and final scale add
// rounding. rtol=2e-5 and atol=2e-7 are diagnostic tolerances for these
// fixtures, not a proven bound on rsqrt error or all finite inputs. Exact
// GPU-vs-GPU bit parity is an independent stricter gate.
constexpr double oracle_atol = 2.0e-7, oracle_rtol = 2.0e-5;
void check(hipError_t error, const char* operation) {
    if (error != hipSuccess) throw std::runtime_error(std::string(operation) + ": hip_error=" + std::to_string(int(error)));
}
std::string digest(const float* data, std::size_t count) {
    unsigned char bytes[SHA256_DIGEST_LENGTH];
    if (!SHA256(reinterpret_cast<const unsigned char*>(data), count * sizeof(float), bytes))
        throw std::runtime_error("sha256-failed");
    constexpr char hex[] = "0123456789abcdef";
    std::string result;
    for (unsigned char b : bytes) { result += hex[b >> 4]; result += hex[b & 15]; }
    return result;
}
struct Resources {
    float *src = nullptr, *bq = nullptr, *bk = nullptr, *cq = nullptr, *ck = nullptr, *scratch = nullptr;
    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr, stop = nullptr;
    hipGraph_t graphs[2]{};
    hipGraphExec_t graph_execs[2]{};
    bool capturing = false;
    std::vector<std::string> cleanup() {
        std::vector<std::string> errors;
        auto record = [&](hipError_t e, const char* op) {
            if (e != hipSuccess) errors.push_back(std::string(op) + ":" + std::to_string(int(e)));
        };
        if (capturing) {
            hipGraph_t abandoned = nullptr;
            record(hipStreamEndCapture(stream, &abandoned), "cleanup-end-capture");
            if (abandoned) record(hipGraphDestroy(abandoned), "cleanup-abandoned-graph");
        }
        if (stream) record(hipStreamSynchronize(stream), "cleanup-stream-sync");
        for (auto e : graph_execs) if (e) record(hipGraphExecDestroy(e), "destroy-graph-exec");
        for (auto g : graphs) if (g) record(hipGraphDestroy(g), "destroy-graph");
        if (stop) record(hipEventDestroy(stop), "destroy-stop");
        if (start) record(hipEventDestroy(start), "destroy-start");
        for (float* p : {bq, bk, cq, ck, scratch, src}) if (p) record(hipFree(p), "free-buffer");
        if (stream) record(hipStreamDestroy(stream), "destroy-stream");
        return errors;
    }
};
struct Case {
    gdn::Shape shape;
    std::string pattern;
    float eps_q = epsilon, eps_k = epsilon;
    std::vector<float> data;
    std::string sha256;
};
std::uint64_t next(std::uint64_t& state) {
    auto z = (state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}
Case make_case(int tokens, int sequences, const std::string& pattern, bool unequal_eps = false) {
    Case c{{width, heads, tokens, sequences}, pattern};
    if (unequal_eps) c.eps_k = 4.0e-6f;
    c.data.assign(std::size_t(tokens) * sequences * token_stride, 17.0f); // unused V lanes
    std::uint64_t seed = UINT64_C(0x67646e6e6f726d31);
    for (int s = 0; s < sequences; ++s) for (int t = 0; t < tokens; ++t)
    for (int which = 0; which < 2; ++which) for (int h = 0; h < heads; ++h)
    for (int col = 0; col < width; ++col) {
        float x = float(int(next(seed) % 2001) - 1000) / 257.0f;
        if (pattern == "zero") x = ((col + h + which) & 1) ? -0.0f : 0.0f;
        else if (pattern == "near-epsilon") x *= std::sqrt(epsilon / width);
        else if (pattern == "tiny") x *= 1.0e-20f;
        else if (pattern == "large") x *= 1.0e15f;
        else if (pattern == "mixed") {
            if (col % 17 == 0) x = -0.0f;
            else if (col % 5 == 0) x *= 1.0e-12f;
            else if (col % 7 == 0) x *= 1.0e6f;
        } else throw std::runtime_error("unknown-input-pattern");
        c.data[(std::size_t(s) * tokens + t) * token_stride + which * k_offset + h * width + col] = x;
    }
    if (c.data.size() > source_capacity || gdn::elements(c.shape) > output_capacity)
        throw std::runtime_error("case-exceeds-capacity");
    c.sha256 = digest(c.data.data(), c.data.size());
    return c;
}
gdn::Input input(const Resources& r, const Case& c, bool k) {
    return {r.src + (k ? k_offset : 0), width, token_stride,
            std::int64_t(token_stride) * c.shape.tokens, k ? c.eps_k : c.eps_q};
}
void launch(Resources& r, const Case& c, bool candidate) {
    const auto q = input(r, c, false), k = input(r, c, true);
    if (candidate) check(gdn::fused(q, k, r.cq, r.ck, c.shape, r.stream), "fused-launch");
    else check(gdn::baseline(q, k, r.bq, r.bk, r.scratch, c.shape, r.stream), "baseline-launch");
}
void prepare(Resources& r, const Case& c) {
    check(hipMemcpy(r.src, c.data.data(), c.data.size() * sizeof(float), hipMemcpyHostToDevice), "upload");
    const auto n = gdn::elements(c.shape);
    for (float* p : {r.bq, r.bk, r.cq, r.ck})
        check(hipMemset(p, 0xff, (n + guard) * sizeof(float)), "poison-output-and-guard");
    check(hipMemset(r.scratch, 0xff, (2 * n + guard) * sizeof(float)), "poison-scratch-and-guard");
}
json validate(Resources& r, const Case& c) {
    const auto n = gdn::elements(c.shape);
    check(hipStreamSynchronize(r.stream), "validation-sync");
    double max_absolute_error = 0, max_scaled_error = 0;
    json hashes = json::array();
    for (int which = 0; which < 2; ++which) {
        std::vector<float> b(n + guard), f(n + guard);
        check(hipMemcpy(b.data(), which ? r.bk : r.bq, b.size()*4, hipMemcpyDeviceToHost), "read-baseline");
        check(hipMemcpy(f.data(), which ? r.ck : r.cq, f.size()*4, hipMemcpyDeviceToHost), "read-candidate");
        if (std::memcmp(b.data(), f.data(), n * sizeof(float))) throw std::runtime_error("GPU-bit-parity-failed");
        for (std::size_t j = n; j < n + guard; ++j) {
            std::uint32_t bb, fb;
            std::memcpy(&bb, &b[j], 4); std::memcpy(&fb, &f[j], 4);
            if (bb != UINT32_MAX || fb != UINT32_MAX) throw std::runtime_error("output-guard-overwrite");
        }
        const double eps = which ? c.eps_k : c.eps_q;
        for (int s = 0; s < c.shape.sequences; ++s) for (int t = 0; t < c.shape.tokens; ++t)
        for (int h = 0; h < heads; ++h) {
            const auto source_row = (std::size_t(s) * c.shape.tokens + t) * token_stride + which * k_offset + h * width;
            const auto output_row = ((std::size_t(s) * c.shape.tokens + t) * heads + h) * width;
            double sum = 0;
            for (int col = 0; col < width; ++col) { const double x = c.data[source_row+col]; sum += x*x; }
            for (int col = 0; col < width; ++col) {
                const double ref = double(c.data[source_row+col]) / std::sqrt(sum + eps);
                const double got = b[output_row+col];
                const double error = std::abs(got-ref);
                const double allowed = oracle_atol + oracle_rtol * std::abs(ref);
                if (!std::isfinite(got) || error > allowed) throw std::runtime_error("float64-oracle-failed");
                max_absolute_error = std::max(max_absolute_error, error);
                max_scaled_error = std::max(max_scaled_error, error/allowed);
            }
        }
        hashes.push_back(digest(b.data(), n));
    }
    std::array<std::uint32_t, guard> scratch_guard{};
    check(hipMemcpy(scratch_guard.data(), r.scratch + 2*n, guard*4, hipMemcpyDeviceToHost), "read-scratch-guard");
    for (auto bits : scratch_guard) if (bits != UINT32_MAX) throw std::runtime_error("scratch-guard-overwrite");
    std::vector<float> reread(c.data.size());
    check(hipMemcpy(reread.data(), r.src, reread.size()*4, hipMemcpyDeviceToHost), "read-input");
    if (std::memcmp(reread.data(), c.data.data(), c.data.size()*4)) throw std::runtime_error("input-mutated");
    return {{"exact_gpu_parity", true}, {"float64_oracle", true}, {"guards_intact", true},
        {"input_unchanged", true}, {"max_absolute_error", max_absolute_error},
        {"max_tolerance_fraction", max_scaled_error}, {"q_k_output_sha256", hashes}};
}
void batch(Resources& r, const Case& c, bool candidate) {
    for (int i = 0; i < launches; ++i) launch(r, c, candidate);
}
void capture_graphs(Resources& r, const Case& c) {
    for (int i = 0; i < 2; ++i) {
        if (r.graph_execs[i]) { check(hipGraphExecDestroy(r.graph_execs[i]), "destroy-prior-exec"); r.graph_execs[i] = nullptr; }
        if (r.graphs[i]) { check(hipGraphDestroy(r.graphs[i]), "destroy-prior-graph"); r.graphs[i] = nullptr; }
        check(hipStreamBeginCapture(r.stream, hipStreamCaptureModeThreadLocal), "begin-capture");
        r.capturing = true;
        batch(r, c, bool(i));
        const auto status = hipStreamEndCapture(r.stream, &r.graphs[i]);
        r.capturing = false;
        check(status, "end-capture");
        check(hipGraphInstantiate(&r.graph_execs[i], r.graphs[i], nullptr, nullptr, 0), "instantiate-graph");
    }
}
void execute_batch(Resources& r, const Case& c, bool candidate, bool graph) {
    if (graph) check(hipGraphLaunch(r.graph_execs[int(candidate)], r.stream), "graph-launch");
    else batch(r, c, candidate);
}
double measure(Resources& r, const Case& c, bool candidate, bool graph) {
    check(hipEventRecord(r.start, r.stream), "record-start");
    execute_batch(r, c, candidate, graph);
    check(hipEventRecord(r.stop, r.stream), "record-stop");
    check(hipEventSynchronize(r.stop), "sync-stop");
    float ms = 0;
    check(hipEventElapsedTime(&ms, r.start, r.stop), "elapsed-time");
    if (!std::isfinite(ms) || ms <= 0) throw std::runtime_error("invalid-event-duration");
    return double(ms) * 1000 / launches;
}
double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return (v[9]+v[10])/2; }
json describe(const Case& c) {
    return {{"width", width}, {"heads", heads}, {"tokens", c.shape.tokens}, {"sequences", c.shape.sequences},
        {"pattern", c.pattern}, {"eps_q", c.eps_q}, {"eps_k", c.eps_k}, {"input_sha256", c.sha256},
        {"source_elements", c.data.size()}, {"stride_head", width}, {"stride_token", token_stride},
        {"stride_sequence", token_stride*c.shape.tokens}, {"k_offset", k_offset}};
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
    std::vector<Case> timed, diagnostic;
    for (int tokens : {1, 16, 512}) timed.push_back(make_case(tokens, 1, "mixed"));
    for (const char* pattern : {"zero", "near-epsilon", "tiny", "large", "mixed"})
        diagnostic.push_back(make_case(3, 2, pattern, true));
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
        {"warp_size", p.warpSize}, {"pci_domain", p.pciDomainID}, {"pci_bus", p.pciBusID}, {"pci_device", p.pciDeviceID}};
    report["hip_runtime_version"] = runtime;
    report["hip_driver_version"] = driver;
    if (std::strncmp(p.gcnArchName, "gfx1151", 7) || (p.gcnArchName[7] && p.gcnArchName[7] != ':'))
        throw std::runtime_error("device-architecture-mismatch");
    // HIP exposes domain/bus/device here; function 0 is checked by launcher preflight.
    if (p.pciDomainID != 0 || p.pciBusID != 0xc2 || p.pciDeviceID != 0)
        throw std::runtime_error("device-pci-mismatch");
    if (p.warpSize != 32) throw std::runtime_error("wave-requires-warp-size-32");
    check(hipStreamCreate(&r.stream), "create-stream");
    check(hipEventCreate(&r.start), "create-start");
    check(hipEventCreate(&r.stop), "create-stop");

    check(hipMalloc(reinterpret_cast<void**>(&r.src), source_capacity * sizeof(float)), "allocate-source");
    for (float** p : {&r.bq, &r.bk, &r.cq, &r.ck})
        check(hipMalloc(reinterpret_cast<void**>(p), (output_capacity + guard)*4), "allocate-output");
    check(hipMalloc(reinterpret_cast<void**>(&r.scratch), (2*output_capacity + guard)*4), "allocate-scratch");
    auto diagnostics = [&](const char* stage) {
        report[stage] = json::array();
        for (const auto& c : diagnostic) {
            report[stage].push_back(describe(c));
            prepare(r, c); launch(r, c, false); launch(r, c, true);
            report[stage].back()["validation"] = validate(r, c);
        }
    };
    diagnostics("diagnostics_before");
    for (const auto& c : timed) {
        report["cases"].push_back(describe(c));
        auto& result = report["cases"].back();
        prepare(r, c); launch(r, c, false); launch(r, c, true);
        result["before"] = validate(r, c);
        result["modes"] = json::array();
        for (bool graph : {false, true}) {
            if (graph) capture_graphs(r, c);
            result["modes"].push_back({{"mode", graph ? "captured_batch_graph" : "direct_dispatch"},
                {"baseline_us_per_call", json::array()}, {"candidate_us_per_call", json::array()}});
            auto& mode = result["modes"].back();
            prepare(r, c); // Detect missing/stale writes independently in each timing mode.
            for (int i = 0; i < warmups; ++i) {
                execute_batch(r, c, false, graph); execute_batch(r, c, true, graph);
            }
            check(hipStreamSynchronize(r.stream), "warmup-sync");
            mode["before"] = validate(r, c);
            std::vector<double> baseline, candidate;
            for (int i = 0; i < samples; ++i) for (bool is_candidate : {bool(i%2), !bool(i%2)}) {
                const double us = measure(r, c, is_candidate, graph);
                (is_candidate ? candidate : baseline).push_back(us);
                mode[is_candidate ? "candidate_us_per_call" : "baseline_us_per_call"].push_back(us);
            }
            mode["after"] = validate(r, c);
            mode["baseline_median_us"] = median(baseline);
            mode["candidate_median_us"] = median(candidate);
        }
    }
    diagnostics("diagnostics_after");
}
} // namespace
int main() {
    Resources resources;
    json report = {{"schema_version", 1}, {"fixture", "rocm10-gdn-norm-compare-v1"},
        {"scope", "source-derived standalone primitive; no full-backend or end-to-end speed claim"},
        {"source_commit", "c7af5c6c29902eb1f7b3bd7952607e2349e1c668"},
        {"generator", "gdn-norm-splitmix64-v1"}, {"seed", UINT64_C(0x67646e6e6f726d31)},
        {"metadata_coverage", "unknown"}, {"vendor_authenticity", "unverified"},
        {"success", false}, {"research_comparison_valid", false},
        {"hip_compile_version", {HIP_VERSION_MAJOR, HIP_VERSION_MINOR, HIP_VERSION_PATCH}},
        {"requested_device_allocation_bytes", allocated_bytes},
        {"external_limits_required", {{"wall_seconds", 30}, {"cpu_seconds", 20}, {"device_bytes", 268435456}}},
        {"warmup_batches_per_provider", warmups}, {"paired_samples", samples}, {"provider_calls_per_batch", launches},
        {"baseline_entry_point", "strixlab_gdn_norm::baseline"}, {"candidate_entry_point", "strixlab_gdn_norm::fused"},
        {"timing_modes", {"direct_dispatch", "captured_batch_graph"}},
        {"graph_scope", "one graph per provider containing 100 fixed-input calls; capture/instantiate excluded; amortized primitive batch, not llama token graph"},
        {"scheduled_kernel_executions", 75065},
        {"baseline_kernels_per_call", 4}, {"candidate_kernels_per_call", 1},
        {"pair_order", "even baseline,candidate; odd candidate,baseline; zero-based"},
        {"timing_scope", "HIP event batch elapsed / 100 provider calls; both q and k including intermediate writes and possible host enqueue gaps; excludes transfers, validation, allocation, event creation"},
        {"semantics", "finite F32 width128 corrected RMS_NORM(eps/128) then SCALE(1/sqrtf(128)); independent q/k epsilon; exact GPU bit parity required"},
        {"oracle", {{"formula", "float64 x/sqrt(sum(x*x)+original_eps)"}, {"atol", oracle_atol}, {"rtol", oracle_rtol}}},
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
    report["research_comparison_valid"] = success;
    try {
        const auto serialized = report.dump(); // Strict UTF-8; never rewrite evidence bytes.
        std::cout << serialized << '\n';
    } catch (const std::exception&) {
        success = false;
        std::cout << "{\"fixture\":\"rocm10-gdn-norm-compare-v1\",\"success\":false,"
                     "\"research_comparison_valid\":false,\"loaded_paths_complete\":false,"
                     "\"failure\":\"json-serialization-failed\"}\n";
    }
    std::cout.flush();
    return success && std::cout.good() ? 0 : 1;
}
