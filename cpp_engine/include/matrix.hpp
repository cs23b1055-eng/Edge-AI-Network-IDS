// Minimal dense row-major matrix + the linear-algebra kernels an MLP needs.
// Written from scratch with the C++ standard library only.
#pragma once
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <algorithm>

#ifndef IDS_REAL
#define IDS_REAL float
#endif
namespace ids {
using Real = IDS_REAL;

struct Matrix {
    size_t rows = 0, cols = 0;
    std::vector<Real> a;
    Matrix() = default;
    Matrix(size_t r, size_t c, Real v = 0) : rows(r), cols(c), a(r * c, v) {}
    Real& at(size_t r, size_t c) { return a[r * cols + c]; }
    Real at(size_t r, size_t c) const { return a[r * cols + c]; }
    Real* row(size_t r) { return a.data() + r * cols; }
    const Real* row(size_t r) const { return a.data() + r * cols; }
    void resize(size_t r, size_t c) { rows = r; cols = c; a.assign(r * c, 0); }
    void zero() { std::fill(a.begin(), a.end(), Real(0)); }
};

// C(m x n) = A(m x k) * B(k x n)           (i-k-j order: contiguous inner loop)
inline void matmul(const Matrix& A, const Matrix& B, Matrix& C) {
    if (A.cols != B.rows) throw std::invalid_argument("matmul: shape mismatch");
    C.resize(A.rows, B.cols);
    for (size_t i = 0; i < A.rows; ++i) {
        Real* c = C.row(i);
        for (size_t k = 0; k < A.cols; ++k) {
            const Real aik = A.at(i, k);
            const Real* b = B.row(k);
            for (size_t j = 0; j < B.cols; ++j) c[j] += aik * b[j];
        }
    }
}
// C(k x n) = A^T * B  where A is (m x k), B is (m x n)    -> weight gradient
inline void matmul_AtB(const Matrix& A, const Matrix& B, Matrix& C) {
    if (A.rows != B.rows) throw std::invalid_argument("matmul_AtB: shape mismatch");
    C.resize(A.cols, B.cols);
    for (size_t m = 0; m < A.rows; ++m) {
        const Real* b = B.row(m);
        for (size_t i = 0; i < A.cols; ++i) {
            const Real ami = A.at(m, i);
            Real* c = C.row(i);
            for (size_t j = 0; j < B.cols; ++j) c[j] += ami * b[j];
        }
    }
}
// C(m x k) = A * B^T where A is (m x n), B is (k x n)     -> back-propagated delta
inline void matmul_ABt(const Matrix& A, const Matrix& B, Matrix& C) {
    if (A.cols != B.cols) throw std::invalid_argument("matmul_ABt: shape mismatch");
    C.resize(A.rows, B.rows);
    for (size_t i = 0; i < A.rows; ++i) {
        const Real* a = A.row(i);
        for (size_t k = 0; k < B.rows; ++k) {
            const Real* b = B.row(k);
            Real s = 0;
            for (size_t j = 0; j < A.cols; ++j) s += a[j] * b[j];
            C.at(i, k) = s;
        }
    }
}
inline void add_row_bias(Matrix& Z, const std::vector<Real>& b) {
    if (Z.cols != b.size()) throw std::invalid_argument("add_row_bias: shape mismatch");
    for (size_t i = 0; i < Z.rows; ++i) { Real* z = Z.row(i); for (size_t j = 0; j < Z.cols; ++j) z[j] += b[j]; }
}
inline void col_sums(const Matrix& A, std::vector<Real>& out) {
    out.assign(A.cols, 0);
    for (size_t i = 0; i < A.rows; ++i) { const Real* r = A.row(i); for (size_t j = 0; j < A.cols; ++j) out[j] += r[j]; }
}
}  // namespace ids
