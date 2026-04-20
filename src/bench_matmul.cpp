#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <string>
#include "matmul.h"
#include "benchmark.h"

void rand_fill(std::vector<float>& v) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : v) x = dist(rng);
}

int main() {
    int sizes[] = {64, 128, 256, 512, 1024};

    std::cout << std::left  << std::setw(28) << "Configuration"
              << std::right << std::setw(12) << "Serial ms"
              << std::setw(12) << "OMP ms"
              << std::setw(10) << "Speedup\n";
    std::cout << std::string(62, '-') << "\n";

    for (int idx = 0; idx < 5; idx++) {
        int N = sizes[idx];
        std::vector<float> A(N * N), B(N * N), C;
        rand_fill(A);
        rand_fill(B);

        // Time the serial run with OpenMP disabled at runtime
        // omp_set_num_threads(1) forces single-threaded execution
        // even in an OpenMP-compiled binary
        double serial_ms = time_ms([&]() {
            // We directly call the triple loop logic here inline
            // to guarantee zero OpenMP involvement
            C.assign(N * N, 0.0f);
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < N; k++)
                        sum += A[i*N+k] * B[k*N+j];
                    C[i*N+j] = sum;
                }
        });

        // Time the OpenMP-parallelized version
        double omp_ms = time_ms([&]() {
            matmul(A, B, C, N, N, N);
        });

        std::string label = std::to_string(N) + "x" + std::to_string(N);
        double speedup = serial_ms / omp_ms;

        std::cout << std::left  << std::setw(28) << label
                  << std::right << std::setw(12) << std::fixed
                  << std::setprecision(3) << serial_ms
                  << std::setw(12) << omp_ms
                  << std::setw(9)  << std::setprecision(2) << speedup << "x\n";
    }

    return 0;
}
