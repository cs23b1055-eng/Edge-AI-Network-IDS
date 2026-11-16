#pragma once
#include <cmath>
#include "matrix.hpp"
namespace ids {
inline void relu_inplace(Matrix& Z) { for (auto& v : Z.a) v = v > 0 ? v : 0; }
// dZ = dA * relu'(Z); A (post-activation) > 0  <=>  Z > 0
inline void relu_backward(const Matrix& A, Matrix& dA) { for (size_t i = 0; i < A.a.size(); ++i) if (A.a[i] <= 0) dA.a[i] = 0; }
// numerically stable row-wise softmax (subtract the row max before exp)
inline void softmax_rows(Matrix& Z) {
    for (size_t i = 0; i < Z.rows; ++i) {
        Real* z = Z.row(i);
        Real mx = z[0]; for (size_t j = 1; j < Z.cols; ++j) mx = std::max(mx, z[j]);
        Real s = 0; for (size_t j = 0; j < Z.cols; ++j) { z[j] = std::exp(z[j] - mx); s += z[j]; }
        for (size_t j = 0; j < Z.cols; ++j) z[j] /= s;
    }
}
}  // namespace ids
