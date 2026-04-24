#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <mpi.h>
#include <omp.h>
#include "matmul_cuda.h"

int main(int argc, char** argv) {
    // --- PHASE 3: MPI INITIALIZATION ---
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            std::cerr << "Error: This hybrid pipeline requires at least 2 MPI ranks.\n";
        }
        MPI_Finalize();
        return 1;
    }

    // Set matrix dimensions for a single transformer layer calculation
    int M = 1024, K = 1024, N = 1024;
    int num_elements = M * N;

    // Allocate memory
    std::vector<float> X(num_elements, 0.5f);  // Input tokens
    std::vector<float> W(num_elements, 0.5f);  // Weights
    std::vector<float> Y(num_elements, 0.0f);  // Output activations

    if (rank == 0) {
        std::cout << "\n======================================================\n";
        std::cout << "Starting UCS645 Hybrid Inference Engine (MPI + OpenMP + CUDA)\n";
        std::cout << "======================================================\n";
        std::cout << "[Rank 0] Initializing Transformer Layers 1-6...\n";
        
        // --- PHASE 4: CUDA GPU OFFLOAD ---
        std::cout << "[Rank 0] Offloading GEMM to RTX 3050...\n";
        matmul_cuda(X, W, Y, M, K, N);

        // --- PHASE 2: OPENMP CPU MULTITHREADING ---
        std::cout << "[Rank 0] Applying ReLU Activation using OpenMP (" << omp_get_max_threads() << " threads)...\n";
        
        #pragma omp parallel for
        for (int i = 0; i < num_elements; i++) {
            // ReLU: max(0, x)
            Y[i] = std::max(0.0f, Y[i]);
        }

        // --- PHASE 3: MPI NETWORK COMMUNICATION ---
        std::cout << "[Rank 0] Computation complete. Sending activations to Rank 1...\n";
        // Send the output Y to Rank 1 (Tag = 0)
        MPI_Send(Y.data(), num_elements, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);

    } else if (rank == 1) {
        std::cout << "[Rank 1] Initializing Transformer Layers 7-12... Waiting for data.\n";

        // --- PHASE 3: MPI NETWORK COMMUNICATION ---
        // Receive the activations from Rank 0 (Tag = 0)
        MPI_Recv(X.data(), num_elements, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "[Rank 1] Activations received from Rank 0.\n";

        // --- PHASE 4: CUDA GPU OFFLOAD ---
        std::cout << "[Rank 1] Offloading GEMM to RTX 3050...\n";
        matmul_cuda(X, W, Y, M, K, N);

        // --- PHASE 2: OPENMP CPU MULTITHREADING ---
        std::cout << "[Rank 1] Applying LayerNorm/Post-processing using OpenMP...\n";
        
        #pragma omp parallel for
        for (int i = 0; i < num_elements; i++) {
            // Dummy post-processing step
            Y[i] = Y[i] * 0.9f; 
        }

        std::cout << "[Rank 1] Transformer Pipeline Complete! Output generated.\n";
        std::cout << "======================================================\n\n";
    }

    // Shut down MPI
    MPI_Finalize();
    return 0;
}
