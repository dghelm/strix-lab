#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

// Mirrors ggml/src/ggml-cuda/argsort.cu at the pinned source:
// k_argsort_f32_i32, lines 167-214. Values are deliberately all equal, so the
// only swaps are the padded-index rules; this is not a replacement kernel.
static std::vector<int> pinned_descending_equal_indices(int n) {
  int padded = 1;
  while (padded < n) padded *= 2;
  std::vector<int> indices(padded);
  for (int i = 0; i < padded; ++i) indices[i] = i;
  for (int k = 2; k <= padded; k *= 2) {
    for (int j = k / 2; j > 0; j /= 2) {
      for (int col = 0; col < padded; ++col) {
        int peer = col ^ j;
        if (peer <= col) continue;
        bool swap = false;
        if ((col & k) == 0)
          swap = indices[col] >= n || (indices[peer] < n && false);
        else
          swap = indices[peer] >= n || (indices[col] < n && false);
        if (swap) std::swap(indices[col], indices[peer]);
      }
    }
  }
  indices.resize(n);
  return indices;
}

int main() {
  const auto result = pinned_descending_equal_indices(11);
  const std::vector<int> expected{0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 9};
  const std::vector<int> expected_top10{0, 1, 2, 3, 4, 5, 6, 7, 10, 8};
  if (result != expected ||
      std::vector<int>(result.begin(), result.begin() + 10) != expected_top10) {
    std::cerr << "pinned bitonic network regression\n";
    return 1;
  }
  return 0;
}
