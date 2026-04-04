#include <iostream>
#include <vector>
#include <cmath>
#include "layernorm.h"

int main() {
    // One row: [1, 2, 3, 4]
    // mean = 2.5, var = 1.25
    // normalized = [-1.34, -0.45, 0.45, 1.34] (approx)
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
// gamma=1, beta=0 means pure normalization, no learned scaling yet
    std::vector<float> gamma = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> beta  = {0.0f, 0.0f, 0.0f, 0.0f};
layernorm(x, gamma, beta, 1, 4);
std::cout << "LayerNorm([1,2,3,4]) = ";
    for (float v : x)
        std::cout << v << " ";
    std::cout << "\n";
// After normalization, mean should be ~0 and variance ~1
    float mean = 0.0f, var = 0.0f;
    for (float v : x) mean += v;
    mean /= 4;
    for (float v : x) var += (v - mean) * (v - mean);
    var /= 4;

    std::cout << "Mean (should be ~0): " << mean << "\n";
    std::cout << "Var  (should be ~1): " << var  << "\n";

    return 0;
}
