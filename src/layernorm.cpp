#include "layernorm.h"
#include <stdexcept>
#include <math.h>
void layernorm(std::vector<float>& x,
               const std::vector<float>& gamma,
               const std::vector<float>& beta,
               int rows, int cols,
               float eps) 
{
	if ((int)gamma.size() != cols || (int)beta.size() != cols)
        throw std::invalid_argument("layernorm: gamma/beta size mismatch");
	for (int i = 0; i < rows; i++) {
	// Step 1: compute mean of this row
        float mean = 0.0f;
        for (int j = 0; j < cols; j++)
            mean += x[i * cols + j];
        mean /= cols;
	// Step 2: compute variance of this row
        float var = 0.0f;
        for (int j = 0; j < cols; j++) {
            float diff = x[i * cols + j] - mean;
            var += diff * diff;
        }
        var /= cols;
	// Step 3: normalize, scale, shift
        float std_inv = 1.0f / sqrtf(var + eps);
	for (int j = 0; j < cols; j++) {
            float norm = (x[i * cols + j] - mean) * std_inv;
         	   x[i * cols + j] = gamma[j] * norm + beta[j];
        }
	}
}
