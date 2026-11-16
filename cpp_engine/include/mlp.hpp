// Multilayer perceptron: Input -> [Dense+ReLU]* -> Dense+Softmax, trained with
// weighted cross-entropy via mini-batch SGD(+momentum) or Adam. No ML framework used.
#pragma once
#include <cstdint>
#include <random>
#include <string>
#include <vector>
#include "matrix.hpp"
namespace ids {

struct Layer {
    Matrix W;                 // (in x out)
    std::vector<Real> b;      // (out)
    // optimizer state
    Matrix mW, vW; std::vector<Real> mb, vb;
};

enum class Optimizer { SGD, ADAM };

struct TrainConfig {
    Real lr = Real(0.001);
    size_t batch = 256;
    size_t epochs = 10;
    Optimizer opt = Optimizer::ADAM;
    Real momentum = Real(0.9);   // SGD only
    Real l2 = 0;                 // weight decay
    uint64_t seed = 42;
};

class Mlp {
public:
    Mlp() = default;
    // sizes = {n_in, h1, h2, ..., n_out}. He-normal init for weights, zero biases.
    Mlp(const std::vector<size_t>& sizes, uint64_t seed);

    const std::vector<size_t>& sizes() const { return sizes_; }
    size_t n_in() const { return sizes_.front(); }
    size_t n_out() const { return sizes_.back(); }
    std::vector<Layer>& layers() { return layers_; }
    const std::vector<Layer>& layers() const { return layers_; }

    // Forward propagation. X is (batch x n_in); returns class probabilities (batch x n_out).
    // Activations are cached in acts_ for backward().
    const Matrix& forward(const Matrix& X);
    // Weighted mean cross-entropy. y = class indices, w = per-class weights (may be empty).
    Real loss(const Matrix& probs, const std::vector<int32_t>& y, const std::vector<Real>& w) const;
    // Backpropagation: fills gW_/gb_ (call after forward on the same X).
    void backward(const Matrix& X, const std::vector<int32_t>& y, const std::vector<Real>& w, Real l2);
    // Apply one optimizer step using the gradients from backward().
    void step(const TrainConfig& cfg);
    // Numerical access for gradient checking
    const std::vector<Matrix>& grad_W() const { return gW_; }
    const std::vector<std::vector<Real>>& grad_b() const { return gb_; }

    // Single-sample / batch inference (no cache mutation of training state needed by callers)
    void predict_proba(const Real* x, size_t n, Real* out_probs);  // n samples, row-major

    bool save(const std::string& path) const;
    static bool load(const std::string& path, Mlp& out, std::string* err = nullptr);

private:
    std::vector<size_t> sizes_;
    std::vector<Layer> layers_;
    std::vector<Matrix> acts_;  // acts_[l] = output of layer l (post-activation)
    std::vector<Matrix> gW_;
    std::vector<std::vector<Real>> gb_;
    uint64_t t_ = 0;  // Adam step counter
};

}  // namespace ids
