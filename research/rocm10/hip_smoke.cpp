// Private diagnostic fixture. Execution requires the reviewed launcher and lease.
#include <hip/hip_runtime.h>
#include <hip/hip_version.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <link.h>

namespace {
constexpr unsigned kElements = 256;
constexpr std::size_t kMaxObjects = 256;
constexpr std::size_t kMaxPath = 4096;

__global__ void fill(std::uint32_t* output) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kElements) output[i] = 3u * i + 1u;
}

struct Evidence {
    std::uint32_t* device = nullptr;
    hipDeviceProp_t properties{};
    int device_count = 0;
    int runtime_version = 0;
    int driver_version = 0;
    const char* failure = nullptr;
    hipError_t hip_error = hipSuccess;
    hipError_t cleanup_error = hipSuccess;
    unsigned checked_elements = 0;
    int mismatch_index = -1;
    std::uint32_t mismatch_actual = 0;
    std::array<std::array<char, kMaxPath>, kMaxObjects> loaded_paths{};
    std::size_t loaded_count = 0;
    bool loaded_overflow = false;
};

bool checked(Evidence& evidence, hipError_t error, const char* operation) {
    if (error == hipSuccess) return true;
    evidence.failure = operation;
    evidence.hip_error = error;
    return false;
}

bool run(Evidence& evidence) {
    if (!checked(evidence, hipRuntimeGetVersion(&evidence.runtime_version), "hipRuntimeGetVersion")) return false;
    if (!checked(evidence, hipDriverGetVersion(&evidence.driver_version), "hipDriverGetVersion")) return false;
    if (!checked(evidence, hipGetDeviceCount(&evidence.device_count), "hipGetDeviceCount")) return false;
    if (evidence.device_count != 1) {
        evidence.failure = "expected-one-visible-device";
        return false;
    }
    if (!checked(evidence, hipSetDevice(0), "hipSetDevice")) return false;
    if (!checked(evidence, hipGetDeviceProperties(&evidence.properties, 0), "hipGetDeviceProperties")) return false;
    const char* architecture = evidence.properties.gcnArchName;
    if (std::strncmp(architecture, "gfx1151", 7) != 0 ||
        (architecture[7] != '\0' && architecture[7] != ':')) {
        evidence.failure = "device-architecture-mismatch";
        return false;
    }
    if (!checked(evidence, hipMalloc(reinterpret_cast<void**>(&evidence.device),
                                    kElements * sizeof(std::uint32_t)), "hipMalloc")) return false;
    if (!checked(evidence, hipMemset(evidence.device, 0, kElements * sizeof(std::uint32_t)), "hipMemset")) return false;
    hipLaunchKernelGGL(fill, dim3(1), dim3(kElements), 0, 0, evidence.device);
    if (!checked(evidence, hipGetLastError(), "kernel-launch")) return false;
    if (!checked(evidence, hipDeviceSynchronize(), "hipDeviceSynchronize")) return false;
    std::array<std::uint32_t, kElements> output{};
    if (!checked(evidence, hipMemcpy(output.data(), evidence.device, sizeof(output),
                                    hipMemcpyDeviceToHost), "hipMemcpy")) return false;
    for (unsigned i = 0; i < kElements; ++i) {
        if (output[i] != 3u * i + 1u) {
            evidence.failure = "output-mismatch";
            evidence.mismatch_index = static_cast<int>(i);
            evidence.mismatch_actual = output[i];
            return false;
        }
        ++evidence.checked_elements;
    }
    return true;
}

int capture_object(dl_phdr_info* object, std::size_t, void* context) {
    auto& evidence = *static_cast<Evidence*>(context);
    const char* name = object->dlpi_name ? object->dlpi_name : "";
    const std::size_t length = strnlen(name, kMaxPath);
    if (evidence.loaded_count == kMaxObjects || length == kMaxPath) {
        evidence.loaded_overflow = true;
        return 1;
    }
    std::memcpy(evidence.loaded_paths[evidence.loaded_count++].data(), name, length + 1);
    return 0;
}

// Encode non-ASCII bytes as escapes so loader/device strings always yield JSON.
void json_string(const char* value, std::size_t maximum) {
    constexpr char hex[] = "0123456789abcdef";
    std::cout << '"';
    for (std::size_t i = 0; i < maximum && value[i]; ++i) {
        const auto byte = static_cast<unsigned char>(value[i]);
        if (byte == '"' || byte == '\\') std::cout << '\\' << static_cast<char>(byte);
        else if (byte < 0x20 || byte >= 0x7f) {
            std::cout << "\\u00" << hex[byte >> 4] << hex[byte & 15];
        } else std::cout << static_cast<char>(byte);
    }
    std::cout << '"';
}

void emit(const Evidence& evidence, bool success) {
    std::cout << "{\"schema_version\":1,\"fixture\":\"rocm10-hip-smoke-v1\",\"success\":"
              << (success ? "true" : "false")
              << ",\"scope\":\"private-diagnostic-only\",\"metadata_coverage\":\"unknown\""
              << ",\"vendor_authenticity\":\"unverified\",\"expected_target\":\"gfx1151\""
              << ",\"device_count\":" << evidence.device_count << ",\"device_ordinal\":0"
              << ",\"device_name\":";
    json_string(evidence.properties.name, sizeof(evidence.properties.name));
    std::cout << ",\"architecture\":";
    json_string(evidence.properties.gcnArchName, sizeof(evidence.properties.gcnArchName));
    std::cout << ",\"pci_domain\":" << evidence.properties.pciDomainID
              << ",\"pci_bus\":" << evidence.properties.pciBusID
              << ",\"pci_device\":" << evidence.properties.pciDeviceID
              << ",\"hip_compile_version\":{\"major\":" << HIP_VERSION_MAJOR
              << ",\"minor\":" << HIP_VERSION_MINOR << ",\"patch\":" << HIP_VERSION_PATCH << '}'
              << ",\"hip_runtime_version\":" << evidence.runtime_version
              << ",\"hip_driver_version\":" << evidence.driver_version
              << ",\"element_count\":" << kElements
              << ",\"checked_elements\":" << evidence.checked_elements
              << ",\"failure\":";
    if (evidence.failure) json_string(evidence.failure, 128); else std::cout << "null";
    std::cout << ",\"hip_error_code\":" << static_cast<int>(evidence.hip_error)
              << ",\"cleanup_error_code\":" << static_cast<int>(evidence.cleanup_error)
              << ",\"mismatch_index\":" << evidence.mismatch_index
              << ",\"mismatch_actual\":" << evidence.mismatch_actual
              << ",\"loaded_paths_encoding\":\"non-ascii-bytes-as-u00xx\""
              << ",\"loaded_paths_complete\":" << (!evidence.loaded_overflow ? "true" : "false")
              << ",\"loaded_paths\":[";
    for (std::size_t i = 0; i < evidence.loaded_count; ++i) {
        if (i) std::cout << ',';
        json_string(evidence.loaded_paths[i].data(), kMaxPath);
    }
    // dl_iterate_phdr reports the main executable as an empty path; preserve it.
    std::cout << "]}\n";
}
} // namespace

int main() {
    static_assert(sizeof(std::uint32_t) == 4);
    Evidence evidence;
    bool success = run(evidence);
    if (evidence.device != nullptr) {
        evidence.cleanup_error = hipFree(evidence.device);
        evidence.device = nullptr;
        if (evidence.cleanup_error != hipSuccess) {
            if (!evidence.failure) evidence.failure = "hipFree";
            success = false;
        }
    }
    if (dl_iterate_phdr(capture_object, &evidence) != 0 || evidence.loaded_overflow) {
        evidence.loaded_overflow = true;
        if (!evidence.failure) evidence.failure = "loaded-object-evidence-limit";
        success = false;
    }
    emit(evidence, success);
    std::cout.flush();
    return success && std::cout.good() ? 0 : 1;
}
