#include <iostream>
#include <vector>
#include <chrono>
#include "matmul.h"
#include "matmul_cuda.h"

int main() {
    // 1024x1024 is large enough to see a massive difference,
    // but small enough that the serial CPU version won't make you wait all day.
    int M = 1024, K = 1024, N = 1024;

    std::cout << "Allocating and filling matrices (" << M << "x" << N << ")...\n";
    std::vector<float> A(M * K, 0.5f);
    std::vector<float> B(K * N, 0.5f);
    std::vector<float> C_cpu, C_gpu;

    // --- GPU WARMUP ---
    // The very first CUDA call always takes extra time because the NVIDIA driver
    // has to initialize the GPU context. We run a tiny dummy multiplication first 
    // so we don't accidentally benchmark the driver startup time.
    std::cout << "Warming up GPU...\n";
    matmul_cuda(A, B, C_gpu, 64, 64, 64);

    // --- BENCHMARK CPU ---
    std::cout << "Running CPU serial matrix multiplication...\n";
    auto start_cpu = std::chrono::high_resolution_clock::now();
    
    matmul(A, B, C_cpu, M, K, N);
    
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;
    std::cout << "CPU Time: " << cpu_time.count() << " ms\n";

    // --- BENCHMARK GPU ---
    std::cout << "Running GPU matrix multiplication...\n";
    auto start_gpu = std::chrono::high_resolution_clock::now();
    
    // Note: This time includes allocating VRAM, copying CPU->GPU, 
    // running the kernel, and copying GPU->CPU. It is an honest "end-to-end" time.
    matmul_cuda(A, B, C_gpu, M, K, N);
    
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_time = end_gpu - start_gpu;
    std::cout << "GPU Time: " << gpu_time.count() << " ms\n";

    // --- RESULTS ---
    double speedup = cpu_time.count() / gpu_time.count();
    std::cout << "\n====================================\n";
    std::cout << "Speedup: " << speedup << "x faster on RTX 3050!" << "\n";
    std::cout << "====================================\n";

    return 0;
}
