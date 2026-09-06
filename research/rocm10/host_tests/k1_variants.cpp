#include "topk_k1_variants.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using Provider = hipError_t (*)(const float*, int*, int, int, hipStream_t);
void require(bool ok) { if (!ok) { std::cerr << "host check failed\n"; std::abort(); } }
void check_case(Provider provider, int columns) {
    // Three rows expose row addressing, repeated ties across waves/strides,
    // negative-only values, and +0/-0 bit patterns with numeric tie semantics.
    constexpr int rows = 3;
    std::vector<float> input(rows * columns + 2, 123456.0f);
    float* src = input.data() + 1;
    for (int c = 0; c < columns; ++c) {
        src[c] = -float(columns - c); // Maximum is last column, always negative.
        src[columns + c] = c % 2 ? -0.0f : 0.0f;
        src[2 * columns + c] = -std::numeric_limits<float>::max();
    }
    // Index31 and index32 cross a wave boundary; index287 crosses a stride.
    for (int index : {31, 32, 255, 256, 287, 1023})
        if (index < columns) src[2 * columns + index] = -1.0f;
    const auto before = input;
    std::vector<int> out(rows + 2, -9876);
    require(provider(src, out.data() + 1, rows, columns, nullptr) == hipSuccess);
    require(host_block_size == ((provider == strixlab_topk_k1_small_hip && columns <= 32) ? 32u : 256u));
    require(out.front() == -9876 && out.back() == -9876);
    require(std::memcmp(before.data(), input.data(), input.size() * sizeof(float)) == 0);
    for (int row = 0; row < rows; ++row) {
        int expected = 0;
        for (int c = 1; c < columns; ++c)
            if (src[row * columns + c] > src[row * columns + expected]) expected = c;
        require(out[row + 1] == expected);
    }
}
int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "wrong-wave") {
        warpSize = 64;
        float input = 1; int out = -1;
        strixlab_topk_k1_wave_hip(&input, &out, 1, 1, nullptr);
        return 0; // Must be unreachable: device assertion must fire in host model.
    }
    for (Provider provider : {strixlab_topk_k1_small_hip, strixlab_topk_k1_wave_hip}) {
        for (int columns : {1, 2, 7, 31, 32, 33, 63, 64, 127, 255, 256, 257, 511, 1023, 1024})
            check_case(provider, columns);
        float input = -1; int output = -1;
        unsigned before = host_launches;
        require(provider(nullptr, &output, 1, 1, nullptr) == hipErrorInvalidValue);
        require(provider(&input, nullptr, 1, 1, nullptr) == hipErrorInvalidValue);
        for (int rows : {-1, 0}) require(provider(&input, &output, rows, 1, nullptr) == hipErrorInvalidValue);
        for (int columns : {-1, 0, 1025}) require(provider(&input, &output, 1, columns, nullptr) == hipErrorInvalidValue);
        require(host_launches == before);
        host_error = 73;
        require(provider(&input, &output, 1, 1, nullptr) == 73);
        host_error = 0;
    }
    std::cout << "PASS: 90 rows, dispatch, canaries, input preservation, ties, signed zeros, invalid lanes, validation and launch errors\n";
}
