#ifndef MATMUL_CUDA_H
#define MATMUL_CUDA_H

#include <vector>

// GPU matrix multiply: C = A * B
// A: (M x K), B: (K x N), C: (M x N)
// Uses shared memory tiling for reduced global memory latency
void matmul_cuda(const std::vector<float>& A,
                 const std::vector<float>& B,
                 std::vector<float>& C,
                 int M, int K, int N);

#endif // MATMUL_CUDA_H
