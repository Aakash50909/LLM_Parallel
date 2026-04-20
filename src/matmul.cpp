#include "matmul.h"
#include <stdexcept>
#include <cmath>

void matmul(const std::vector<float>& A,
            const std::vector<float>& B,
            std::vector<float>& C,
            int M, int K, int N) {

    if ((int)A.size() != M * K)
        throw std::invalid_argument("matmul: A size mismatch");
    if ((int)B.size() != K * N)
        throw std::invalid_argument("matmul: B size mismatch");

    C.assign(M * N, 0.0f);

    // #pragma omp parallel for tells OpenMP to split the outer loop
    // across all available threads automatically.
    // Each thread gets a unique value of i and works on its own
    // rows of C — no two threads ever write to the same memory location
    // so there are zero race conditions here.
    // schedule(static) divides iterations into equal chunks upfront —
    // best for uniform work like matmul where every row costs the same.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}
