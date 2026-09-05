#include "capsule_transport.hpp"
#include "third_party/nlohmann/json.hpp"

#include <array>
#include <charconv>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <openssl/evp.h>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace strixlab::topk {
namespace {
using Json = nlohmann::json;
constexpr const char *identity = "strixlab-topk-host-test-v1";
constexpr std::size_t max_request = 1024 * 1024;
constexpr int required_seals =
    F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE;
void require(bool condition) {
  if (!condition)
    throw std::runtime_error("invalid native fixture request");
}
std::string canonical(const Json &value) { return value.dump(2) + "\n"; }
class Hash {
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx{EVP_MD_CTX_new(),
                                                              EVP_MD_CTX_free};

public:
  Hash() {
    require(ctx && EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) == 1);
  }
  void update(const char *data, std::size_t length) {
    require(EVP_DigestUpdate(ctx.get(), data, length) == 1);
  }
  std::string finish() {
    unsigned char bytes[EVP_MAX_MD_SIZE];
    unsigned length;
    require(EVP_DigestFinal_ex(ctx.get(), bytes, &length) == 1 && length == 32);
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (unsigned i = 0; i < length; ++i)
      out << std::setw(2) << unsigned(bytes[i]);
    return out.str();
  }
};
std::string sha(const std::string &bytes) {
  Hash hash;
  hash.update(bytes.data(), bytes.size());
  return hash.finish();
}
std::string executable_sha() {
  std::ifstream file("/proc/self/exe", std::ios::binary);
  require(file.is_open());
  Hash hash;
  std::array<char, 65536> buffer;
  while (file.read(buffer.data(), buffer.size()) || file.gcount())
    hash.update(buffer.data(), static_cast<std::size_t>(file.gcount()));
  require(file.eof() && !file.bad());
  return hash.finish();
}
std::string read_request(const std::string &path) {
  constexpr const char *prefix = "/proc/self/fd/";
  require(path.rfind(prefix, 0) == 0);
  auto number = path.substr(std::char_traits<char>::length(prefix));
  int fd = -1;
  auto parsed =
      std::from_chars(number.data(), number.data() + number.size(), fd);
  require(parsed.ec == std::errc{} &&
          parsed.ptr == number.data() + number.size() && fd >= 0 &&
          number == std::to_string(fd));
  const int flags = fcntl(fd, F_GETFL);
  require(flags >= 0 && (flags & O_ACCMODE) == O_RDONLY);
  require(fcntl(fd, F_GET_SEALS) == required_seals);
  struct stat before{}, after{};
  require(fstat(fd, &before) == 0 && S_ISREG(before.st_mode) &&
          before.st_size > 0 &&
          static_cast<std::uint64_t>(before.st_size) <= max_request);
  std::string bytes(static_cast<std::size_t>(before.st_size), '\0');
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    // The Python runner may leave the inherited file offset at EOF.
    auto count = pread(fd, bytes.data() + offset, bytes.size() - offset,
                       static_cast<off_t>(offset));
    if (count < 0 && errno == EINTR)
      continue;
    require(count > 0);
    offset += static_cast<std::size_t>(count);
  }
  require(fstat(fd, &after) == 0 && before.st_dev == after.st_dev &&
          before.st_ino == after.st_ino && before.st_size == after.st_size &&
          before.st_mode == after.st_mode &&
          before.st_mtim.tv_sec == after.st_mtim.tv_sec &&
          before.st_mtim.tv_nsec == after.st_mtim.tv_nsec &&
          before.st_ctim.tv_sec == after.st_ctim.tv_sec &&
          before.st_ctim.tv_nsec == after.st_ctim.tv_nsec);
  return bytes;
}
Json parse_request(const std::string &bytes) {
  std::vector<std::set<std::string>> keys;
  auto callback = [&](int depth, Json::parse_event_t event, Json &value) {
    require(depth <= 16);
    if (event == Json::parse_event_t::object_start)
      keys.emplace_back();
    else if (event == Json::parse_event_t::key)
      require(!keys.empty() &&
              keys.back().insert(value.get<std::string>()).second);
    else if (event == Json::parse_event_t::object_end)
      keys.pop_back();
    return true;
  };
  auto request = Json::parse(bytes, callback);
  require(request.is_object());
  return request;
}
bool hash_string(const Json &value) {
  if (!value.is_string())
    return false;
  const auto &text = value.get_ref<const std::string &>();
  return text.size() == 64 &&
         text.find_first_not_of("0123456789abcdef") == std::string::npos;
}
Json scenario() {
  Json comparison{{"policy", "paired-latency-log-bootstrap-v1"},
                  {"protected_regression_bps", nullptr},
                  {"permitted_arm_differences", Json::array({"candidate-id"})}};
  Json coordinates = Json::array();
  const auto digest = input_digest(host_fixture::input());
  unsigned order = 0;
  for (const auto *mode : {"fixture-direct", "fixture-replay"}) {
    coordinates.push_back({{"coordinate_id", mode},
                           {"case_id", "host-tie-fixture"},
                           {"case_set", "evaluation"},
                           {"mode", mode},
                           {"order", order++},
                           {"input_id", "host-tie-fixture"},
                           {"input_sha256", digest},
                           {"warmup_count", 1},
                           {"sample_count", 5}});
  }
  return {{"schema_version", 1},
          {"comparison", comparison},
          {"coordinates", coordinates}};
}
void validate_request(const Json &request, const std::string &operation,
                      const std::string &bytes, const Json &expected_scenario) {
  const std::set<std::string> fields{"schema_version",
                                     "protocol",
                                     "operation",
                                     "capsule_id",
                                     "candidate",
                                     "scenario_sha256",
                                     "manifest_sha256",
                                     "executable_sha256",
                                     "prior_response_sha256",
                                     "scenario_contract_sha256",
                                     "scenario"};
  require(request.size() == fields.size());
  for (const auto &item : request.items())
    require(fields.count(item.key()) == 1);
  require(request.at("schema_version").is_number_integer() &&
          request.at("schema_version") == 1);
  require(request.at("protocol") == "native-capsule-v1" &&
          request.at("operation") == operation);
  require(request.at("capsule_id") == identity);
  const auto &candidate = request.at("candidate");
  require(candidate == "host-fixture" || candidate == "baseline-hip" ||
          candidate == "rocprim-topk" || candidate == "rocprim-segmented-topk");
  require(request.at("scenario_sha256") == sha(identity));
  require(hash_string(request.at("manifest_sha256")) &&
          hash_string(request.at("executable_sha256")));
  require(request.at("executable_sha256") == executable_sha());
  if (operation == "describe") {
    require(request.at("prior_response_sha256").is_null() &&
            request.at("scenario_contract_sha256").is_null() &&
            request.at("scenario").is_null());
  } else {
    require(hash_string(request.at("prior_response_sha256")));
    // Exact serialization also rejects numerical type coercions such as
    // order=0.0.
    require(canonical(request.at("scenario")) == canonical(expected_scenario));
    require(request.at("scenario_contract_sha256") ==
            sha(canonical(expected_scenario)));
  }
  // All accepted fields are ASCII strings, nulls and bounded integers. No
  // generic floating-point serializer equivalence is assumed for arbitrary JSON
  // payloads.
  require(canonical(request) == bytes);
}
Json binding(const Json &request, const std::string &request_bytes) {
  auto response = request;
  response.erase("scenario");
  response["request_sha256"] = sha(request_bytes);
  response["opaque_payload"] = {{"fixture", true}, {"synthetic", true}};
  return response;
}
} // namespace
int capsule_main(int argc, char **argv, host_fixture::Fault fault) {
  try {
    require(argc == 4 && std::string(argv[2]) == "--request");
    std::string operation = argv[1];
    require(operation == "describe" || operation == "correctness" ||
            operation == "benchmark");
    auto bytes = read_request(argv[3]);
    auto request = parse_request(bytes);
    auto contract = scenario();
    validate_request(request, operation, bytes, contract);
    auto response = binding(request, bytes);
    if (operation == "describe")
      response["scenario"] = contract;
    else {
      // This is recomputed in *each process*. The prior SHA is not a readiness
      // token.
      auto gate = host_fixture::check(fault, operation == "benchmark",
                                      request["candidate"] == "host-fixture");
      response["opaque_payload"]["reason"] = gate.reason;
      response["opaque_payload"]["readiness_checks"] = gate.readiness_checks;
      response["opaque_payload"]["setup_calls"] = gate.setup_calls;
      response["opaque_payload"]["operation_calls"] = gate.operation_calls;
      if (operation == "benchmark" && !gate.passed) {
        std::cerr << "native host fixture benchmark readiness failed\n";
        return 2;
      }
      auto coordinates = Json::array();
      for (const auto &coordinate : contract["coordinates"]) {
        if (operation == "correctness")
          coordinates.push_back(
              {{"coordinate", coordinate}, {"passed", gate.passed}});
        else
          // Explicit test data, not CPU timings or a HIP/graph performance
          // claim.
          coordinates.push_back(
              {{"coordinate", coordinate},
               {"workspace_bytes", 0},
               {"latency_seconds",
                Json::array({0.001, 0.002, 0.003, 0.004, 0.005})}});
      }
      response["coordinates"] = coordinates;
    }
    auto output = canonical(response);
    require(output.size() <= 1024 * 1024);
    std::cout << output;
    return std::cout.good() ? 0 : 2;
  } catch (const std::exception &) {
    // Never reflect untrusted request bytes or parser exception text into logs.
    std::cerr << "native host fixture request rejected\n";
    return 2;
  }
}
} // namespace strixlab::topk
