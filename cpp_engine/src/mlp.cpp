#include "mlp.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include "activations.hpp"
namespace ids {

Mlp::Mlp(const std::vector<size_t>& sizes, uint64_t seed) : sizes_(sizes) {
    if (sizes.size() < 2) throw std::invalid_argument("Mlp needs at least input and output sizes");
    std::mt19937_64 rng(seed);
    for (size_t l = 0; l + 1 < sizes.size(); ++l) {
        Layer L;
        const size_t in = sizes[l], out = sizes[l + 1];
        L.W.resize(in, out);
        // He initialisation: N(0, sqrt(2/fan_in)) -- suited to ReLU
        std::normal_distribution<double> nd(0.0, std::sqrt(2.0 / double(in)));
        for (auto& w : L.W.a) w = Real(nd(rng));
        L.b.assign(out, 0);
        L.mW.resize(in, out); L.vW.resize(in, out); L.mb.assign(out, 0); L.vb.assign(out, 0);
        layers_.push_back(std::move(L));
    }
    acts_.resize(layers_.size());
}

const Matrix& Mlp::forward(const Matrix& X) {
    const Matrix* in = &X;
    for (size_t l = 0; l < layers_.size(); ++l) {
        matmul(*in, layers_[l].W, acts_[l]);
        add_row_bias(acts_[l], layers_[l].b);
        if (l + 1 < layers_.size()) relu_inplace(acts_[l]); else softmax_rows(acts_[l]);
        in = &acts_[l];
    }
    return acts_.back();
}

Real Mlp::loss(const Matrix& P, const std::vector<int32_t>& y, const std::vector<Real>& w) const {
    double s = 0;
    for (size_t i = 0; i < P.rows; ++i) {
        const double p = std::max<double>(P.at(i, size_t(y[i])), 1e-12);
        s += (w.empty() ? 1.0 : double(w[size_t(y[i])])) * -std::log(p);
    }
    return Real(s / double(P.rows));
}

void Mlp::backward(const Matrix& X, const std::vector<int32_t>& y, const std::vector<Real>& w, Real l2) {
    const size_t L = layers_.size(), B = X.rows;
    gW_.resize(L); gb_.resize(L);
    // Output layer: for softmax + cross-entropy, dL/dz = (p - onehot(y)) * w_y / B
    Matrix delta = acts_.back();
    for (size_t i = 0; i < B; ++i) {
        const Real wi = w.empty() ? Real(1) : w[size_t(y[i])];
        Real* d = delta.row(i);
        d[y[i]] -= 1;
        for (size_t j = 0; j < delta.cols; ++j) d[j] *= wi / Real(B);
    }
    for (size_t l = L; l-- > 0;) {
        const Matrix& in = (l == 0) ? X : acts_[l - 1];
        matmul_AtB(in, delta, gW_[l]);                 // dW = a_{l-1}^T * delta
        col_sums(delta, gb_[l]);                       // db = sum over batch
        if (l2 > 0) for (size_t i = 0; i < gW_[l].a.size(); ++i) gW_[l].a[i] += l2 * layers_[l].W.a[i];
        if (l > 0) {
            Matrix nd;
            matmul_ABt(delta, layers_[l].W, nd);       // delta_{l-1} = delta * W^T
            relu_backward(acts_[l - 1], nd);           // * relu'
            delta = std::move(nd);
        }
    }
}

void Mlp::step(const TrainConfig& c) {
    ++t_;
    const double b1 = 0.9, b2 = 0.999, eps = 1e-8;
    const double c1 = 1.0 - std::pow(b1, double(t_)), c2 = 1.0 - std::pow(b2, double(t_));
    for (size_t l = 0; l < layers_.size(); ++l) {
        Layer& L = layers_[l];
        auto upd = [&](Real* p, Real* m, Real* v, const Real* g, size_t n) {
            for (size_t i = 0; i < n; ++i) {
                if (c.opt == Optimizer::SGD) { m[i] = c.momentum * m[i] - c.lr * g[i]; p[i] += m[i]; }
                else {
                    m[i] = Real(b1 * m[i] + (1 - b1) * g[i]);
                    v[i] = Real(b2 * v[i] + (1 - b2) * g[i] * g[i]);
                    p[i] -= Real(c.lr * (m[i] / c1) / (std::sqrt(double(v[i]) / c2) + eps));
                }
            }
        };
        upd(L.W.a.data(), L.mW.a.data(), L.vW.a.data(), gW_[l].a.data(), L.W.a.size());
        upd(L.b.data(), L.mb.data(), L.vb.data(), gb_[l].data(), L.b.size());
    }
}

void Mlp::predict_proba(const Real* x, size_t n, Real* out) {
    Matrix X(n, n_in());
    std::memcpy(X.a.data(), x, n * n_in() * sizeof(Real));
    const Matrix& P = forward(X);
    std::memcpy(out, P.a.data(), n * n_out() * sizeof(Real));
}

// Binary model file: "IDSMLP01" | u32 n_sizes | u64 sizes[] | per layer: W (in*out) f32, b (out) f32
bool Mlp::save(const std::string& path) const {
    FILE* f = std::fopen(path.c_str(), "wb"); if (!f) return false;
    std::fwrite("IDSMLP01", 1, 8, f);
    uint32_t n = uint32_t(sizes_.size()); std::fwrite(&n, 4, 1, f);
    for (auto s : sizes_) { uint64_t v = s; std::fwrite(&v, 8, 1, f); }
    for (auto& L : layers_) {
        for (auto w : L.W.a) { float v = float(w); std::fwrite(&v, 4, 1, f); }
        for (auto b : L.b) { float v = float(b); std::fwrite(&v, 4, 1, f); }
    }
    return std::fclose(f) == 0;
}

bool Mlp::load(const std::string& path, Mlp& out, std::string* err) {
    auto fail = [&](const char* m) { if (err) *err = m; return false; };
    FILE* f = std::fopen(path.c_str(), "rb"); if (!f) return fail("cannot open file");
    char magic[8]; uint32_t n = 0;
    if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, "IDSMLP01", 8) != 0) { std::fclose(f); return fail("bad magic"); }
    if (std::fread(&n, 4, 1, f) != 1 || n < 2 || n > 64) { std::fclose(f); return fail("bad layer count"); }
    std::vector<size_t> sizes(n);
    for (auto& s : sizes) { uint64_t v; if (std::fread(&v, 8, 1, f) != 1 || v == 0 || v > 100000) { std::fclose(f); return fail("bad size"); } s = size_t(v); }
    Mlp m(sizes, 0);
    for (auto& L : m.layers_) {
        for (auto& w : L.W.a) { float v; if (std::fread(&v, 4, 1, f) != 1) { std::fclose(f); return fail("truncated"); } w = Real(v); }
        for (auto& b : L.b) { float v; if (std::fread(&v, 4, 1, f) != 1) { std::fclose(f); return fail("truncated"); } b = Real(v); }
    }
    std::fclose(f);
    out = std::move(m);
    return true;
}
}  // namespace ids
