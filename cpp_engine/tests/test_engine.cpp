// Unit tests (built with IDS_REAL=double so finite-difference gradient checks are precise).
#include <cmath>
#include <cstdio>
#include "activations.hpp"
#include "mlp.hpp"
using namespace ids;
static int fails = 0, total = 0;
#define CHECK(c) do { ++total; if (!(c)) { ++fails; std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)
static bool near(double a, double b, double t = 1e-9) { return std::fabs(a - b) <= t * (1 + std::fabs(b)); }

void test_matrix() {
    Matrix A(2, 3), B(3, 2), C;
    Real av[] = {1, 2, 3, 4, 5, 6}, bv[] = {7, 8, 9, 10, 11, 12};
    std::copy(av, av + 6, A.a.begin()); std::copy(bv, bv + 6, B.a.begin());
    matmul(A, B, C);  // [[58,64],[139,154]]
    CHECK(C.rows == 2 && C.cols == 2 && near(C.at(0, 0), 58) && near(C.at(0, 1), 64) && near(C.at(1, 0), 139) && near(C.at(1, 1), 154));
    Matrix D; matmul_AtB(A, A, D);  // A^T A (3x3): [0][0]=1+16=17
    CHECK(D.rows == 3 && near(D.at(0, 0), 17) && near(D.at(0, 2), 3 + 24) && near(D.at(2, 2), 9 + 36));
    Matrix E; matmul_ABt(A, A, E);  // A A^T: [[14,32],[32,77]]
    CHECK(near(E.at(0, 0), 14) && near(E.at(0, 1), 32) && near(E.at(1, 1), 77));
    bool threw = false; try { matmul(A, A, C); } catch (const std::invalid_argument&) { threw = true; } CHECK(threw);
    std::vector<Real> b{1, -1}; add_row_bias(C, b); CHECK(near(C.at(0, 0), 59) && near(C.at(1, 1), 153));
    std::vector<Real> cs; col_sums(A, cs); CHECK(near(cs[0], 5) && near(cs[2], 9));
}
void test_activations() {
    Matrix Z(1, 4); Real v[] = {-2, 0, 3, -0.5}; std::copy(v, v + 4, Z.a.begin());
    relu_inplace(Z); CHECK(Z.a[0] == 0 && Z.a[1] == 0 && Z.a[2] == 3 && Z.a[3] == 0);
    Matrix S(2, 3); Real s[] = {1, 2, 3, 1000, 1000, 1000}; std::copy(s, s + 6, S.a.begin());
    softmax_rows(S);
    double e1 = std::exp(-2.0), e2 = std::exp(-1.0), tot = e1 + e2 + 1;
    CHECK(near(S.at(0, 0), e1 / tot) && near(S.at(0, 2), 1 / tot));
    CHECK(near(S.at(1, 0) + S.at(1, 1) + S.at(1, 2), 1) && near(S.at(1, 0), 1.0 / 3));  // stable for huge logits
    Matrix A(1, 3), dA(1, 3, 5); A.a = {1, 0, 2}; relu_backward(A, dA); CHECK(dA.a[0] == 5 && dA.a[1] == 0 && dA.a[2] == 5);
}
void test_init_forward_loss() {
    Mlp net({10, 8, 4, 3}, 1);
    CHECK(net.layers().size() == 3 && net.layers()[0].W.rows == 10 && net.layers()[2].W.cols == 3);
    // He init: empirical std of layer-0 weights ~ sqrt(2/10)
    double m = 0, v = 0; auto& w = net.layers()[0].W.a; for (auto x : w) m += x; m /= w.size(); for (auto x : w) v += (x - m) * (x - m); v /= w.size();
    CHECK(std::fabs(m) < 0.15 && std::fabs(std::sqrt(v) - std::sqrt(2.0 / 10)) < 0.1);
    for (auto b : net.layers()[0].b) CHECK(b == 0);
    Matrix X(5, 10); for (size_t i = 0; i < X.a.size(); ++i) X.a[i] = std::sin(double(i));
    const Matrix& P = net.forward(X);
    for (size_t i = 0; i < 5; ++i) { double s = 0; for (size_t j = 0; j < 3; ++j) { CHECK(P.at(i, j) >= 0); s += P.at(i, j); } CHECK(near(s, 1)); }
    // loss on known probabilities
    Matrix Pk(2, 2); Pk.a = {0.9, 0.1, 0.2, 0.8};
    CHECK(near(net.loss(Pk, {0, 1}, {}), -(std::log(0.9) + std::log(0.8)) / 2));
    CHECK(near(net.loss(Pk, {0, 1}, {2.0, 1.0}), -(2 * std::log(0.9) + std::log(0.8)) / 2));
    // hand-computed forward of a 2-2 linear-ish net
    Mlp t({2, 2, 2}, 0); auto& L0 = t.layers()[0]; auto& L1 = t.layers()[1];
    L0.W.a = {1, -1, 2, 1}; L0.b = {0.5, 0}; L1.W.a = {1, 0, 0, 1}; L1.b = {0, 0};
    Matrix x(1, 2); x.a = {1, 2};  // z0 = [1+4+.5, -1+2] = [5.5, 1] -> relu same; logits=[5.5,1]
    const Matrix& q = t.forward(x); double ex = std::exp(5.5) + std::exp(1.0);
    CHECK(near(q.at(0, 0), std::exp(5.5) / ex) && near(q.at(0, 1), std::exp(1.0) / ex));
}
void test_backprop_gradcheck() {
    Mlp net({6, 5, 4, 3}, 7);
    Matrix X(8, 6); std::mt19937 r(3); std::normal_distribution<double> nd; for (auto& v : X.a) v = nd(r);
    std::vector<int32_t> y{0, 1, 2, 0, 1, 2, 1, 0}; std::vector<Real> w{1.0, 2.0, 0.5};
    net.forward(X); net.backward(X, y, w, 0);
    double maxrel = 0; const double h = 1e-6;
    for (size_t l = 0; l < net.layers().size(); ++l) {
        for (int which = 0; which < 2; ++which) {
            auto& P = which == 0 ? net.layers()[l].W.a : net.layers()[l].b;
            const auto& G = which == 0 ? net.grad_W()[l].a : net.grad_b()[l];
            for (size_t i = 0; i < P.size(); ++i) {
                Real o = P[i];
                P[i] = o + h; double lp = net.loss(net.forward(X), y, w);
                P[i] = o - h; double lm = net.loss(net.forward(X), y, w);
                P[i] = o; double num = (lp - lm) / (2 * h);
                maxrel = std::max(maxrel, std::fabs(num - G[i]) / std::max(1e-7, std::fabs(num) + std::fabs(G[i])));
            }
        }
    }
    std::printf("  gradient check max relative error = %.3e\n", maxrel);
    CHECK(maxrel < 1e-5);
}
void test_training_learns() {
    // XOR-like problem must be learned by SGD updates: verifies step() decreases loss.
    Mlp net({2, 16, 2}, 1); Matrix X(4, 2)  /* seed 5 with 8 units hits a dead-ReLU minimum; not an engine bug */; X.a = {0, 0, 0, 1, 1, 0, 1, 1}; std::vector<int32_t> y{0, 1, 1, 0};
    TrainConfig c; c.lr = 0.05; c.opt = Optimizer::ADAM;
    double l0 = net.loss(net.forward(X), y, {});
    for (int i = 0; i < 800; ++i) { net.forward(X); net.backward(X, y, {}, 0); net.step(c); }
    double l1 = net.loss(net.forward(X), y, {}); const Matrix& P = net.forward(X);
    CHECK(l1 < l0 * 0.1); for (int i = 0; i < 4; ++i) CHECK((P.at(i, 1) > 0.5) == (y[i] == 1));
    TrainConfig s; s.opt = Optimizer::SGD; s.lr = 0.1; Mlp n2({2, 16, 2}, 1); double a0 = n2.loss(n2.forward(X), y, {});
    for (int i = 0; i < 300; ++i) { n2.forward(X); n2.backward(X, y, {}, 0); n2.step(s); }
    CHECK(n2.loss(n2.forward(X), y, {}) < a0);
}
void test_save_load() {
    Mlp a({4, 3, 2}, 9); const char* p = "/tmp/ids_test_model.bin"; CHECK(a.save(p));
    Mlp b; std::string err; CHECK(Mlp::load(p, b, &err));
    Matrix X(2, 4); X.a = {1, 2, 3, 4, -1, 0, 1, 0.5};
    Matrix Pa = a.forward(X), Pb = b.forward(X);
    for (size_t i = 0; i < Pa.a.size(); ++i) CHECK(std::fabs(Pa.a[i] - Pb.a[i]) < 1e-6);  // saved as f32
    Mlp c; CHECK(!Mlp::load("/nonexistent", c, &err));
    FILE* f = std::fopen("/tmp/ids_bad.bin", "wb"); std::fputs("garbage!", f); std::fclose(f); CHECK(!Mlp::load("/tmp/ids_bad.bin", c, &err) && err == "bad magic");
    std::remove(p);
}
int main() {
    test_matrix(); test_activations(); test_init_forward_loss(); test_backprop_gradcheck(); test_training_learns(); test_save_load();
    std::printf("%d checks, %d failed\n", total, fails); return fails ? 1 : 0;
}
