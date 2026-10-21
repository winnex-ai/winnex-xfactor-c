// test_spectral.cpp — validates the spectral log-phase filter in isolation.
//
// Gate C6: the OpenCL `spectral_phase` kernel must match the analytic reference
//   F(k) = exp(i·α·arctan(ln(k+ε))) applied to a complex (re,im) pair:
//     re' = re·cos(φ) − im·sin(φ)
//     im' = re·sin(φ) + im·cos(φ)      with φ = α·arctan(ln(k+ε))
// Also: α=0 → identity; deterministic (same input → same output).
#include "winnex_xfactor_c/attention_forward.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using winnex_xfactor_c::HilbertAttention;

int main() {
    int failures = 0;

    // Geometry (small, matching a head block).
    const int n_heads = 4, n_kv = 2, hd = 8;
    const int d_q = n_heads * hd, d_kv = n_kv * hd, d = 32;
    const double alpha = 0.3;

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // Vector of complex interleaved pairs [2·n].
    const int n = 64;
    std::vector<float> v(2 * n);
    for (auto& x : v) x = nd(rng);

    HilbertAttention attn(d_q, d_kv, n_heads, n_kv, hd, d);

    // 1. apply_spectral with α=0 → identity (the filter is a no-op).
    {
        std::vector<float> v0 = v;
        attn.apply_spectral(v0.data(), n, 0.0);
        bool ok = true;
        for (int i = 0; i < 2 * n; ++i)
            if (std::fabs(v0[i] - v[i]) > 1e-5) ok = false;
        printf("[α=0] apply_spectral identity: %s\n", ok ? "OK" : "FAIL");
        if (!ok) ++failures;
    }

    // 2. apply_spectral with α=0.3 matches the analytic formula elementwise.
    {
        std::vector<float> got = v;
        attn.apply_spectral(got.data(), n, alpha);
        bool ok = true;
        double maxe = 0.0;
        for (int i = 0; i < n; ++i) {
            float phi = (float)alpha * atan(log((float)(i + 1) + 1e-10f));
            float c = cos(phi), s = sin(phi);
            float re = v[2*i], im = v[2*i+1];
            float re2 = re * c - im * s, im2 = re * s + im * c;
            double e = std::fabs((double)got[2*i] - re2) + std::fabs((double)got[2*i+1] - im2);
            if (e > maxe) maxe = e;
            if (e > 1e-4) ok = false;
        }
        printf("[α=0.3] kernel == analytic formula (max|Δ| = %.2e): %s\n",
               maxe, ok ? "OK" : "FAIL");
        if (!ok) ++failures;
    }

    // 3. The filter is a rotation: it preserves the norm of each pair.
    {
        std::vector<float> got = v;
        attn.apply_spectral(got.data(), n, alpha);
        bool ok = true;
        double maxe = 0.0;
        for (int i = 0; i < n; ++i) {
            float ni = std::sqrt(v[2*i]*v[2*i] + v[2*i+1]*v[2*i+1]);
            float no = std::sqrt(got[2*i]*got[2*i] + got[2*i+1]*got[2*i+1]);
            double dn = std::fabs(ni - no);
            if (dn > maxe) maxe = dn;
            if (dn > 1e-4) ok = false;
        }
        printf("[α=0.3] rotation preserves norm (max|Δnorm| = %.2e): %s\n",
               maxe, ok ? "OK" : "FAIL");
        if (!ok) ++failures;
    }

    if (failures == 0) {
        printf("\nALL SPECTRAL TESTS PASSED\n");
        return 0;
    }
    printf("\n%d FAILURE(S)\n", failures);
    return 1;
}
