#include "matmul_cuda.h"
#include <cuda_runtime.h>
#include <stdexcept>

// TILE_SIZE defines the dimensions of the shared memory tile
// 16x16 = 256 threads per block, which is a good occupancy target
// for the RTX 3050. Each thread computes one output element.
#define TILE_SIZE 16

// __global__ marks this as a GPU kernel — called from CPU, runs on GPU
// Every thread executes this function simultaneously
// Each thread is responsible for computing exactly one element of C
__global__ void matMulKernel(const float* A, const float* B, float* C,
                              int M, int K, int N) {

    // Shared memory tiles — declared __shared__ so they live in
    // the fast on-chip SRAM shared by all threads in this block.
    // Each block loads a TILE_SIZE x TILE_SIZE chunk of A and B
    // into shared memory, does the partial dot products, then
    // loads the next tile. This dramatically reduces global memory reads.
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // threadIdx: this thread's position within its block (0..TILE_SIZE-1)
    // blockIdx:  this block's position within the grid
    // blockDim:  size of each block (TILE_SIZE x TILE_SIZE)
    int tx = threadIdx.x;  // column within tile
    int ty = threadIdx.y;  // row within tile

    // Global row and column this thread is responsible for in C
    // This maps (block position * tile size + thread position) to
    // the actual row/col in the full output matrix
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    // Accumulator for the dot product — stays in a register
    // (fastest memory on the GPU, private to each thread)
    float sum = 0.0f;

    // Number of tiles needed to cover the K dimension
    // We step through A left-to-right and B top-to-bottom in tiles
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {

        // Collaboratively load tile of A into shared memory
        // Thread (ty, tx) loads element (row, t*TILE_SIZE + tx) of A
        // Each thread loads exactly one element — fully parallel load
        int aCol = t * TILE_SIZE + tx;
        if (row < M && aCol < K)
            tileA[ty][tx] = A[row * K + aCol];
        else
            tileA[ty][tx] = 0.0f;  // zero-pad out-of-bounds

        // Collaboratively load tile of B into shared memory
        // Thread (ty, tx) loads element (t*TILE_SIZE + ty, col) of B
        int bRow = t * TILE_SIZE + ty;
        if (bRow < K && col < N)
            tileB[ty][tx] = B[bRow * N + col];
        else
            tileB[ty][tx] = 0.0f;  // zero-pad out-of-bounds

        // __syncthreads() is a barrier — ALL threads in the block must
        // reach this point before any thread proceeds.
        // Without this, some threads might start computing before
        // others have finished loading their tile elements.
        __syncthreads();

        // Each thread computes its partial dot product using the tile
        // All reads come from fast shared memory, not slow global memory
        for (int k = 0; k < TILE_SIZE; k++)
            sum += tileA[ty][k] * tileB[k][tx];

        // Second barrier — prevent any thread from overwriting the tile
        // with the next iteration's data before all threads finish using
        // the current tile's values
        __syncthreads();
    }

    // Write the final result to global memory
    // Guard against out-of-bounds writes for non-square matrices
    if (row < M && col < N)
        C[row * N + col] = sum;
}

void matmul_cuda(const std::vector<float>& A,
                 const std::vector<float>& B,
                 std::vector<float>& C,
                 int M, int K, int N) {

    // Step 1: allocate device (GPU) memory
    // cudaMalloc works like malloc but on the GPU's VRAM
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Step 2: copy input data from CPU RAM to GPU VRAM
    // cudaMemcpy direction: cudaMemcpyHostToDevice = CPU -> GPU
    cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Step 3: configure the launch geometry
    // Each block is TILE_SIZE x TILE_SIZE threads
    dim3 blockDim(TILE_SIZE, TILE_SIZE);

    // Grid dimensions: how many blocks needed to cover the output matrix
    // Ceiling division ensures we cover non-multiples of TILE_SIZE
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);

    // Step 4: launch the kernel
    // Triple angle brackets <<<gridDim, blockDim>>> is CUDA launch syntax
    // This is not standard C++ — only nvcc understands it
    matMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);

    // Step 5: wait for GPU to finish
    // Kernel launches are asynchronous — CPU continues immediately after
    // cudaDeviceSynchronize blocks the CPU until all GPU work is done
    cudaDeviceSynchronize();

    // Step 6: copy result back from GPU VRAM to CPU RAM
    C.resize(M * N);
    cudaMemcpy(C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Step 7: free GPU memory — same discipline as free() in C
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
