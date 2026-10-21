// test_derive_preserves.cpp — validates the adaptive weight derivation.
//
// Gate F1:  ‖W_q'·W_z − W_q‖_F (relative) < 1e-3
//   Proves that the latent coupling is an EXACT reparametrization of the
//   classical QKV projections (adaptability without training).
//
// Covers both dense (d_q == d_kv) and GQA (d_q != d_kv) shapes, matching
// Qwen3-4B-DFlash (d_q=4096, d_kv=1024, d=2560).
#include "winnex_xfactor_c/derive_weights.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using winnex_xfactor_c::DerivedWeights;
using winnex_xfactor_c::derive_weights;
using winnex_xfactor_c::preservation_error;

namespace {

// Random weight matrix [out × in], N(0, 1/sqrt(in)) — realistic scale.
std::vector<float> rand_weight(std::mt19937& rng, int out, int in) {
    std::normal_distribution<float> nd(0.0f, 1.0f / std::sqrt((float)in));
    std::vector<float> W((size_t)out * in);
    for (auto& v : W) v = nd(rng);
    return W;
}

bool run_case(const char* name, int d, int d_q, int d_kv, unsigned seed) {
    std::mt19937 rng(seed);
    auto W_q = rand_weight(rng, d_q, d);
    auto W_k = rand_weight(rng, d_kv, d);
    auto W_v = rand_weight(rng, d_kv, d);

    DerivedWeights dw = derive_weights(W_q.data(), W_k.data(), W_v.data(),
                                       d, d_q, d_kv, seed);

    // Shape checks.
    bool shape_ok =
        dw.W_z.size() == (size_t)4 * d * d &&
        dw.W_q_prime.size() == (size_t)d_q * 4 * d &&
        dw.W_r_prime.size() == (size_t)d_kv * 4 * d &&
        dw.W_h_prime.size() == (size_t)d_kv * 4 * d;
    if (!shape_ok) {
        printf("  [%s] SHAPE FAIL: W_z=%zu (want %zu), W_q'=%zu (want %zu), "
               "W_r'=%zu (want %zu), W_h'=%zu (want %zu)\n",
               name, dw.W_z.size(), (size_t)4*d*d, dw.W_q_prime.size(),
               (size_t)d_q*4*d, dw.W_r_prime.size(), (size_t)d_kv*4*d,
               dw.W_h_prime.size(), (size_t)d_kv*4*d);
        return false;
    }

    // Preservation: ‖W_q'·W_z − W_q‖_F / ‖W_q‖_F.
    double err_q = preservation_error(dw, W_q.data());
    bool ok = (err_q >= 0.0 && err_q < 1e-3);
    printf("  [%s] preservation ‖W_q'·W_z − W_q‖_F rel = %.3e  [%s]\n",
           name, err_q, ok ? "OK" : "FAIL");
    return ok;
}

} // namespace

int main() {
    int failures = 0;

    printf("=== F1: adaptive weight derivation (QKV → latent coupling) ===\n");

    // Case 1: dense (d_q == d_kv), small.
    if (!run_case("dense d=16 q=16 kv=16", 16, 16, 16, 1)) ++failures;

    // Case 2: GQA-like (d_q = 4·d_kv, matching Qwen3 ratio 4096/1024).
    if (!run_case("GQA d=64 q=256 kv=64", 64, 256, 64, 2)) ++failures;

    // Case 3: larger dense (closer to real hidden=2560, but small for CI).
    if (!run_case("dense d=128 q=128 kv=128", 128, 128, 128, 3)) ++failures;

    // Case 4: different seed (determinism/robustness).
    if (!run_case("GQA alt seed d=96 q=384 kv=96", 96, 384, 96, 7)) ++failures;

    // Determinism: same seed → identical W_z.
    {
        std::mt19937 rng(5);
        auto W_q = rand_weight(rng, 32, 64);
        auto W_k = rand_weight(rng, 16, 64);
        auto W_v = rand_weight(rng, 16, 64);
        auto a = derive_weights(W_q.data(), W_k.data(), W_v.data(), 64, 32, 16, 99);
        auto b = derive_weights(W_q.data(), W_k.data(), W_v.data(), 64, 32, 16, 99);
        bool same = (a.W_z == b.W_z) && (a.W_q_prime == b.W_q_prime) &&
                    (a.W_r_prime == b.W_r_prime) && (a.W_h_prime == b.W_h_prime);
        printf("  [determinism] same seed → identical weights: %s\n",
               same ? "OK" : "FAIL");
        if (!same) ++failures;
    }

    if (failures == 0) {
        printf("\nALL DERIVE-WEIGHTS TESTS PASSED\n");
        return 0;
    }
    printf("\n%d FAILURE(S)\n", failures);
    return 1;
}
