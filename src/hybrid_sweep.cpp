#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <mpi.h>
#include <omp.h>
#include "matmul.h"       // CPU Baseline
#include "matmul_cuda.h"  // GPU Compute

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Matrix sizes to test
    std::vector<int> sizes = {128, 256, 512, 1024, 2048};

    if (rank == 0) {
        std::cout << "Matrix_Size,Serial_Time_ms(x2),Hybrid_Time_ms,Speedup\n";
    }

    // Warmup GPU
    std::vector<float> X_w(64 * 64, 0.5f), W_w(64 * 64, 0.5f), Y_w(64 * 64, 0.0f);
    matmul_cuda(X_w, W_w, Y_w, 64, 64, 64);
    MPI_Barrier(MPI_COMM_WORLD);

    for (int N : sizes) {
        int num_elements = N * N;
        std::vector<float> X(num_elements, 0.5f);
        std::vector<float> W(num_elements, 0.5f);
        std::vector<float> Y_hybrid(num_elements, 0.0f);
        std::vector<float> Y_serial(num_elements, 0.0f);

        double serial_time_ms = 0.0;

        // --- DYNAMIC SERIAL BASELINE (Rank 0 computes this live) ---
        if (rank == 0) {
            auto start_cpu = std::chrono::high_resolution_clock::now();
            matmul(X, W, Y_serial, N, N, N); // Calculate 1 block sequentially
            auto end_cpu = std::chrono::high_resolution_clock::now();
            serial_time_ms = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
        }

        // Synchronize all ranks before starting the hybrid timer
        MPI_Barrier(MPI_COMM_WORLD);
        double start_time = MPI_Wtime();

        // --- PIPELINE EXECUTION ---
        int num_batches = 8; 
        
        for (int b = 0; b < num_batches; ++b) {
            if (rank == 0) {
                // Rank 0: Layer 1-6 Compute
                matmul_cuda(X, W, Y_hybrid, N, N, N);
                
                #pragma omp parallel for
                for (int i = 0; i < num_elements; i++) {
                    Y_hybrid[i] = std::max(0.0f, Y_hybrid[i]);
                }
                
                // Send to Rank 1
                MPI_Send(Y_hybrid.data(), num_elements, MPI_FLOAT, 1, b, MPI_COMM_WORLD);
                
            } else if (rank == 1) {
                // Rank 1: Wait for data from Rank 0
                MPI_Recv(X.data(), num_elements, MPI_FLOAT, 0, b, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                // Rank 1: Layer 7-12 Compute
                matmul_cuda(X, W, Y_hybrid, N, N, N);
                
                #pragma omp parallel for
                for (int i = 0; i < num_elements; i++) {
                    Y_hybrid[i] = Y_hybrid[i] * 0.9f; 
                }
            }
        }

        // Wait for pipeline to completely finish
        MPI_Barrier(MPI_COMM_WORLD);
        double end_time = MPI_Wtime();

        double total_hybrid_time_ms = (end_time - start_time) * 1000.0;
        double avg_hybrid_time_ms = total_hybrid_time_ms / num_batches;

        // Print results only from Rank 0
        if (rank == 0) {
            // The hybrid pipeline processes 2 blocks across the nodes.
            // To compare fairly, we multiply the single-block serial time by 2.
            double equivalent_serial_time = serial_time_ms * 2;
            double speedup = equivalent_serial_time / avg_hybrid_time_ms;
            
            std::cout << N << "," 
                      << equivalent_serial_time << "," 
                      << avg_hybrid_time_ms << "," 
                      << speedup << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
