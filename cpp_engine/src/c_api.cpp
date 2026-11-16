// Flat C interface (for Python ctypes). All functions are exception-safe; errors -> negative codes.
#include <cstring>
#include <string>
#include "mlp.hpp"
using namespace ids;
extern "C" {
struct IdsModel { Mlp net; };
static thread_local std::string g_err;
const char* ids_last_error() { return g_err.c_str(); }
void* ids_load(const char* path) {
    try { auto* m = new IdsModel; if (!Mlp::load(path, m->net, &g_err)) { delete m; return nullptr; } return m; }
    catch (const std::exception& e) { g_err = e.what(); return nullptr; }
}
void ids_free(void* h) { delete static_cast<IdsModel*>(h); }
int ids_n_in(void* h) { return h ? int(static_cast<IdsModel*>(h)->net.n_in()) : -1; }
int ids_n_out(void* h) { return h ? int(static_cast<IdsModel*>(h)->net.n_out()) : -1; }
// x: n*n_in floats (row-major); out: n*n_out floats. Returns 0 on success.
int ids_predict_proba(void* h, const float* x, int n, float* out) {
    if (!h || !x || !out || n <= 0) { g_err = "invalid argument"; return -1; }
    try { static_cast<IdsModel*>(h)->net.predict_proba(reinterpret_cast<const Real*>(x), size_t(n), reinterpret_cast<Real*>(out)); return 0; }
    catch (const std::exception& e) { g_err = e.what(); return -2; }
}
int ids_real_is_float() { return sizeof(Real) == sizeof(float); }
}
