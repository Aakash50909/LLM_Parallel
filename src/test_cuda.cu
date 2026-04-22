#include <iostream>
#include <vector>
#include <cmath>
#include "matmul.h"
#include "matmul_cuda.h"

int main() {
    // Use a large enough matrix that tiling actually kicks in
    int M = 64, K = 64, N = 64;

    std::vector<float> A(M * K), B(K * N), C_cpu, C_gpu;

    // Fill with simple values so we can verify
    for (int i = 0; i < M * K; i++) A[i] = (float)(i % 7) * 0.1f;
    for (int i = 0; i < K * N; i++) B[i] = (float)(i % 5) * 0.1f;

    // Run both versions
    matmul(A, B, C_cpu, M, K, N);
    matmul_cuda(A, B, C_gpu, M, K, N);

    // Compare results — should be identical within floating point tolerance
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; i++)
        max_diff = std::max(max_diff, std::abs(C_cpu[i] - C_gpu[i]));

    std::cout << "Max difference CPU vs GPU: " << max_diff << "\n";
    std::cout << "CUDA matmul test " << (max_diff < 1e-3f ? "PASSED" : "FAILED") << "\n";

    return 0;
}
