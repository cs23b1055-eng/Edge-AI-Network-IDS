// CLI trainer.  Data files (written by preprocessing/build_dataset.py):
//   X: int32 n, int32 d, float32[n*d]    y: int32 n, int32[n]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <sstream>
#include "mlp.hpp"
using namespace ids;
struct Data { size_t n = 0, d = 0; std::vector<float> X; std::vector<int32_t> y; };
static bool read_data(const std::string& xp, const std::string& yp, Data& D) {
    FILE* f = std::fopen(xp.c_str(), "rb"); if (!f) return false;
    int32_t n, d; if (std::fread(&n, 4, 1, f) != 1 || std::fread(&d, 4, 1, f) != 1) return false;
    D.n = n; D.d = d; D.X.resize(size_t(n) * d);
    if (std::fread(D.X.data(), 4, D.X.size(), f) != D.X.size()) return false; std::fclose(f);
    f = std::fopen(yp.c_str(), "rb"); if (!f) return false;
    int32_t m; if (std::fread(&m, 4, 1, f) != 1 || m != n) return false;
    D.y.resize(n); if (std::fread(D.y.data(), 4, n, f) != size_t(n)) return false; std::fclose(f); return true;
}
static void gather(const Data& D, const std::vector<size_t>& idx, size_t s, size_t e, Matrix& X, std::vector<int32_t>& y) {
    X.resize(e - s, D.d); y.resize(e - s);
    for (size_t i = s; i < e; ++i) { std::memcpy(X.row(i - s), &D.X[idx[i] * D.d], D.d * sizeof(float)); y[i - s] = D.y[idx[i]]; }
}
struct Metrics { double loss, acc, macro_f1; std::vector<double> recall; };
static Metrics evaluate(Mlp& net, const Data& D, size_t K, const std::vector<Real>& w) {
    std::vector<std::vector<size_t>> cm(K, std::vector<size_t>(K, 0));
    double ls = 0; const size_t B = 4096; Matrix X; std::vector<int32_t> y; std::vector<size_t> idx(D.n); std::iota(idx.begin(), idx.end(), 0);
    for (size_t s = 0; s < D.n; s += B) {
        size_t e = std::min(D.n, s + B); gather(D, idx, s, e, X, y);
        const Matrix& P = net.forward(X); ls += net.loss(P, y, {}) * double(e - s);
        for (size_t i = 0; i < e - s; ++i) { size_t a = 0; for (size_t j = 1; j < K; ++j) if (P.at(i, j) > P.at(i, a)) a = j; cm[size_t(y[i])][a]++; }
    }
    Metrics m; m.loss = ls / D.n; size_t ok = 0; double f1s = 0; int cnt = 0; m.recall.assign(K, 0);
    for (size_t c = 0; c < K; ++c) {
        size_t tp = cm[c][c], fn = 0, fp = 0; for (size_t j = 0; j < K; ++j) { if (j != c) { fn += cm[c][j]; fp += cm[j][c]; } }
        ok += tp; if (tp + fn == 0) continue;
        double p = tp + fp ? double(tp) / (tp + fp) : 0, r = double(tp) / (tp + fn);
        m.recall[c] = r; f1s += (p + r) ? 2 * p * r / (p + r) : 0; ++cnt;
    }
    m.acc = double(ok) / D.n; m.macro_f1 = f1s / std::max(cnt, 1); return m;
}
int main(int argc, char** argv) {
    std::map<std::string, std::string> a;
    for (int i = 1; i + 1 < argc; i += 2) a[argv[i]] = argv[i + 1];
    auto get = [&](const char* k, const char* def) { return a.count(k) ? a[k] : std::string(def); };
    if (!a.count("--train-x")) { std::fprintf(stderr, "usage: ids_train --train-x X --train-y Y --val-x X --val-y Y --classes K --out model.bin [--hidden 64,32 --lr 0.001 --batch 256 --epochs 10 --opt adam|sgd --cw-power 0.5 --seed 42 --l2 0 --patience 4]\n"); return 2; }
    Data tr, va;
    if (!read_data(a["--train-x"], a["--train-y"], tr) || !read_data(a["--val-x"], a["--val-y"], va)) { std::fprintf(stderr, "failed to read data\n"); return 1; }
    const size_t K = std::stoul(get("--classes", "2"));
    TrainConfig cfg; cfg.lr = Real(std::stod(get("--lr", "0.001"))); cfg.batch = std::stoul(get("--batch", "256"));
    cfg.epochs = std::stoul(get("--epochs", "10")); cfg.opt = get("--opt", "adam") == "sgd" ? Optimizer::SGD : Optimizer::ADAM;
    cfg.l2 = Real(std::stod(get("--l2", "0"))); cfg.seed = std::stoull(get("--seed", "42")); const int patience = std::stoi(get("--patience", "4"));
    std::vector<size_t> sizes{tr.d}; { std::stringstream ss(get("--hidden", "64,32")); std::string t; while (std::getline(ss, t, ',')) if (!t.empty()) sizes.push_back(std::stoul(t)); } sizes.push_back(K);
    // class weights ~ (N / (K * n_c))^power  (power 0 = unweighted)
    std::vector<size_t> cnt(K, 0); for (auto v : tr.y) cnt[size_t(v)]++;
    const double pw = std::stod(get("--cw-power", "0.5")); std::vector<Real> w(K, 1);
    for (size_t c = 0; c < K; ++c) w[c] = cnt[c] ? Real(std::pow(double(tr.n) / (K * cnt[c]), pw)) : 1;
    Mlp net(sizes, cfg.seed), best; double best_f1 = -1; int bad = 0;
    std::mt19937_64 rng(cfg.seed); std::vector<size_t> idx(tr.n); std::iota(idx.begin(), idx.end(), 0);
    Matrix X; std::vector<int32_t> y; auto t0 = std::chrono::steady_clock::now();
    for (size_t ep = 1; ep <= cfg.epochs; ++ep) {
        std::shuffle(idx.begin(), idx.end(), rng); double tl = 0; size_t nb = 0;
        for (size_t s = 0; s < tr.n; s += cfg.batch) {
            size_t e = std::min(tr.n, s + cfg.batch); gather(tr, idx, s, e, X, y);
            const Matrix& P = net.forward(X); tl += net.loss(P, y, w); ++nb;
            net.backward(X, y, w, cfg.l2); net.step(cfg);
        }
        Metrics m = evaluate(net, va, K, w);
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("{\"epoch\":%zu,\"train_loss\":%.6f,\"val_loss\":%.6f,\"val_acc\":%.6f,\"val_macro_f1\":%.6f,\"elapsed_s\":%.1f}\n", ep, tl / nb, m.loss, m.acc, m.macro_f1, el);
        std::fflush(stdout);
        if (m.macro_f1 > best_f1) { best_f1 = m.macro_f1; best = net; bad = 0; } else if (++bad >= patience) break;
    }
    if (!best.save(get("--out", "model.bin"))) { std::fprintf(stderr, "save failed\n"); return 1; }
    std::printf("{\"best_val_macro_f1\":%.6f}\n", best_f1); return 0;
}
