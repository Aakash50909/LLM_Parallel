#include "attention.h"
#include "matmul.h"
#include "softmax.h"
#include <cmath>
#include <vector>
void attention(const std::vector<float>& Q,
               const std::vector<float>& K,
               const std::vector<float>& V,
               std::vector<float>& out,
               int seq_len, int d_k, int d_v) 
{
// Step 1: scores = Q * Kᵀ
    // Q is (seq_len x d_k), Kᵀ is (d_k x seq_len)
    // scores is (seq_len x seq_len)
    // But matmul expects B in normal layout, so we transpose K manually
    std::vector<float> Kt(d_k * seq_len);
	for (int i = 0; i < seq_len; i++)
        for (int j = 0; j < d_k; j++)
            Kt[j * seq_len + i] = K[i * d_k + j];
	// scores = Q * Kᵀ  shape: (seq_len x seq_len)
    std::vector<float> scores;
    matmul(Q, Kt, scores, seq_len, d_k, seq_len);
	// Step 2: scale by 1/sqrt(d_k)
    float scale = 1.0f / sqrtf((float)d_k);
    for (float& s : scores)
        s *= scale;
	// Step 3: softmax over each row of scores
    // Each row is one query token attending to all key tokens
    softmax(scores, seq_len, seq_len);
	// Step 4: out = scores * V
    // scores: (seq_len x seq_len), V: (seq_len x d_v)
    // out:    (seq_len x d_v)
    matmul(scores, V, out, seq_len, seq_len, d_v);
}

