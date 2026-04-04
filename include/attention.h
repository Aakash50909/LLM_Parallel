#ifndef ATTENTION_H
#define ATTENTION_H

#include <vector>
// Single-head scaled dot-product attention
// Q: (seq_len x d_k)  — queries
// K: (seq_len x d_k)  — keys
// V: (seq_len x d_v)  — values
// out: (seq_len x d_v) — result
// All matrices flat row-major
void attention(const std::vector<float>& Q,
               const std::vector<float>& K,
               const std::vector<float>& V,
               std::vector<float>& out,
               int seq_len, int d_k, int d_v);
#endif // ATTENTION_H
