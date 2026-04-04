#include <iostream>
#include <vector>
#include <cmath>
#include "attention.h"

int main() {
    // 3 tokens, d_k = 4, d_v = 4
    int seq_len = 3, d_k = 4, d_v = 4;
// Simple Q, K, V — identity-like so we can reason about output
    std::vector<float> Q = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0
    };
    std::vector<float> K = Q; // same as Q for this test
    std::vector<float> V = {
        1, 0, 0, 0,
        0, 2, 0, 0,
        0, 0, 3, 0
    };
std::vector<float> out;
    attention(Q, K, V, out, seq_len, d_k, d_v);
std::cout << "Attention output (" << seq_len << " x " << d_v << "):\n";
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_v; j++)
            std::cout << out[i * d_v + j] << " ";
        std::cout << "\n";
    }
// Check output shape is correct
    bool shape_ok = ((int)out.size() == seq_len * d_v);
    std::cout << "Shape check (should be " << seq_len * d_v << "): "
              << out.size() << " — "
              << (shape_ok ? "PASSED" : "FAILED") << "\n";

    return 0;
}
