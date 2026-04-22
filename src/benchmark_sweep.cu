#include <iostream>
#include <vector>
#include <chrono>
#include "matmul.h"
#include "matmul_cuda.h"

int main() {
    // Sizes to test. 
    // We stop at 2048 because the CPU version will take nearly a minute.
    std::vector<int> sizes = {128, 256, 512, 1024, 2048};

    // --- GPU WARMUP ---
    std::vector<float> A_w(64 * 64, 0.5f), B_w(64 * 64, 0.5f), C_w;
    matmul_cuda(A_w, B_w, C_w, 64, 64, 64);

    // Print CSV Header
    std::cout << "Matrix_Size,CPU_Time_ms,GPU_Time_ms,Speedup\n";

    for (int N : sizes) {
        std::vector<float> A(N * N, 0.5f);
        std::vector<float> B(N * N, 0.5f);
        std::vector<float> C_cpu, C_gpu;

        // Benchmark CPU
        auto start_cpu = std::chrono::high_resolution_clock::now();
        matmul(A, B, C_cpu, N, N, N);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        double cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

        // Benchmark GPU
        auto start_gpu = std::chrono::high_resolution_clock::now();
        matmul_cuda(A, B, C_gpu, N, N, N);
        auto end_gpu = std::chrono::high_resolution_clock::now();
        double gpu_time = std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();

        // Calculate and print metrics for this size
        double speedup = cpu_time / gpu_time;
        std::cout << N << "," << cpu_time << "," << gpu_time << "," << speedup << "\n";
    }

    return 0;
}
