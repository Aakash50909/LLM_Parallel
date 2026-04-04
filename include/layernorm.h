#ifndef LAYERNORM_H
#define LAYERNORM_H
#include <vector>
// Applies Layer Normalization to each row of x
// x:     input matrix  (rows x cols), modified in-place
// gamma: scale params  (cols,) — learned, init to 1.0
// beta:  shift params  (cols,) — learned, init to 0.0
// eps:   small constant for numerical stability (default 1e-5)
void layernorm(std::vector<float>& x,
               const std::vector<float>& gamma,
               const std::vector<float>& beta,
               int rows, int cols,
               float eps = 1e-5f);
#endif
