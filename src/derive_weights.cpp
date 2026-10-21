// derive_weights.cpp — adaptive weight derivation (QKV → latent coupling).
#include "winnex_xfactor_c/derive_weights.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace winnex_xfactor_c {

namespace {

// Deterministic 4×4 orthonormal block (a unit quaternion rotation R(q)).
// R is the SO(4) left-multiplication matrix by a unit quaternion q:
//   R(q)·[w,x,y,z]ᵀ = q ⊗ [w,x,y,z]
// This is orthogonal (R·Rᵀ = I) by the norm-preservation of ℍ.
void rotation_block(float* R, const double q[4]) {
    const double w = q[0], x = q[1], y = q[2], z = q[3];
    const double n2 = w*w + x*x + y*y + z*z;
    const double inv = (n2 < 1e-12) ? 1.0 : 1.0 / std::sqrt(n2);
    const double a = w * inv, b = x * inv, c = y * inv, d = z * inv;
    // Row-major [4×4]: R[i][j] = (q ⊗ e_j)_i
    R[0*4+0] = (float)a;  R[0*4+1] = (float)(-b); R[0*4+2] = (float)(-c); R[0*4+3] = (float)(-d);
    R[1*4+0] = (float)b;  R[1*4+1] = (float)a;   R[1*4+2] = (float)(-d); R[1*4+3] = (float)c;
    R[2*4+0] = (float)c;  R[2*4+1] = (float)d;   R[2*4+2] = (float)a;   R[2*4+3] = (float)(-b);
    R[3*4+0] = (float)d;  R[3*4+1] = (float)(-c); R[3*4+2] = (float)b;  R[3*4+3] = (float)a;
}

} // namespace

DerivedWeights derive_weights(const float* W_q, const float* W_k,
                              const float* W_v, int d, int d_q, int d_kv,
                              unsigned seed) {
    if (!W_q || !W_k || !W_v || d <= 0 || d_q <= 0 || d_kv <= 0) {
        throw std::runtime_error("derive_weights: invalid dimensions");
    }

    DerivedWeights dw;
    dw.d = d; dw.d_q = d_q; dw.d_kv = d_kv;

    // ── Step 1: build W_z = [ I ; R₁ ; R₂ ; R₃ ] ∈ ℝ^{4d×d} ──────────────
    // Each 4×4 block of W_z is an orthonormal rotation. We tile the SAME
    // rotation pattern across every 4-wide group of rows, deterministic in
    // `seed`. Because each block is orthonormal (RᵢᵀRᵢ = I), the LEFT-inverse
    // relation holds:  W_zᵀ·W_z = I + R₁ᵀR₁ + R₂ᵀR₂ + R₃ᵀR₃ = 4·I_d
    // (W_z is a TALL 4d×d matrix, so the simple identity is on W_zᵀ·W_z).
    dw.W_z.assign((size_t)4 * d * d, 0.0f);

    // Three rotation quaternions, deterministic from the seed (LCG).
    auto mix = [&](int idx) {
        unsigned s = seed + (unsigned)idx * 2654435761u;
        s ^= s >> 13; s *= 0x5bd1e995u; s ^= s >> 15;
        return (double)(s % 1000000u) / 1000000.0;  // [0,1)
    };
    double q[3][4];
    for (int i = 0; i < 3; ++i) {
        // Unit quaternion: w = cos(θ/2), axis from two angles.
        double theta = 2.0 * 3.141592653589793 * mix(3*i+0);
        double omega = 2.0 * 3.141592653589793 * mix(3*i+1);
        double phi   = 2.0 * 3.141592653589793 * mix(3*i+2);
        q[i][0] = std::cos(theta / 2.0);
        q[i][1] = std::sin(theta / 2.0) * std::cos(omega);
        q[i][2] = std::sin(theta / 2.0) * std::sin(omega) * std::cos(phi);
        q[i][3] = std::sin(theta / 2.0) * std::sin(omega) * std::sin(phi);
    }

    // Block i covers rows [i*d, (i+1)*d). Row r ↦ block (r mod 4) of that group.
    double q_identity[4] = {1.0, 0.0, 0.0, 0.0};
    for (int i = 0; i < 4; ++i) {
        const double* rot = (i == 0) ? q_identity : q[i-1];
        float R[16];
        rotation_block(R, rot);
        const size_t row0 = (size_t)i * d;
        for (int r = 0; r < d; ++r) {
            const int b = r % 4;                 // which 4-row group inside the block
            const int grp = (r - b) / 4;         // which group
            const size_t out_row = row0 + (size_t)r;
            // W_z[out_row][col]: col within the group = g, col = grp*4 + g
            for (int g = 0; g < 4; ++g) {
                const int col = grp * 4 + g;
                if (col < d) dw.W_z[out_row * (size_t)d + col] = R[b*4 + g];
            }
        }
    }

    // ── Step 2: W' = (1/4)·W·W_zᵀ  (pseudoinverse) ────────────────────────
    // W_zᵀ is [d × 4d]; W is [out × d]. W·W_zᵀ = [out × 4d].
    auto derive = [&](const float* W, int out, std::vector<float>& Wp) {
        Wp.assign((size_t)out * 4 * d, 0.0f);
#pragma omp parallel for schedule(static)
        for (int o = 0; o < out; ++o) {
            const float* wrow = W + (size_t)o * d;
            float* prow = Wp.data() + (size_t)o * (4 * d);
            for (int z = 0; z < 4 * d; ++z) {
                // W'[o][z] = Σ_i W[o][i] · W_zᵀ[i][z] = Σ_i W[o][i] · W_z[z][i]
                // W_z is [4d × d]: element (row=z, col=i) at z*d + i.
                float s = 0.0f;
                const float* wzrow = dw.W_z.data() + (size_t)z * d;
                for (int i = 0; i < d; ++i) s += wrow[i] * wzrow[i];
                prow[z] = 0.25f * s;
            }
        }
    };

    derive(W_q, d_q, dw.W_q_prime);
    derive(W_k, d_kv, dw.W_r_prime);
    derive(W_v, d_kv, dw.W_h_prime);

    return dw;
}

double preservation_error(const DerivedWeights& dw, const float* W_q,
                          int max_rows) {
    const int d = dw.d, d_q = dw.d_q;
    if (!W_q) return -1.0;
    if (dw.W_z.size() != (size_t)4 * d * d) return -1.0;
    if (dw.W_q_prime.size() != (size_t)d_q * 4 * d) return -1.0;

    const int rows = (max_rows > 0) ? std::min(max_rows, d_q) : d_q;

    // Reconstruct W_q_hat = W_q'·W_z  [rows × d]  and compare to the original.
    //   W_q_hat[o][i] = Σ_z W_q'[o][z] · W_z[z][i]
    // W_z is [4d × d]: element (row=z, col=i) at z*d + i.
    // Per-row partial sums (num, den) reduced by OpenMP when available.
    //
    // PORTABILITY FIX (2026-10-21). This block called omp_get_thread_num()
    // unconditionally while `#include <omp.h>` is inside `#ifdef _OPENMP`.
    // Apple clang ships without OpenMP by default, so the macOS wheel build
    // fails with:
    //     derive_weights.cpp:133:25: error: use of undeclared identifier
    //     'omp_get_thread_num'
    // (reproduced in CI: Wheels on macos-14, run #1).
    //
    // The pragmas are ignored by a compiler without OpenMP, so the serial
    // fallback below is the whole computation in one thread. The result is
    // identical — OpenMP only splits the row loop.
    const int max_threads = 256;
    std::vector<double> partial_num(max_threads, 0.0), partial_den(max_threads, 0.0);
#ifdef _OPENMP
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
#else
    {
        const int tid = 0;
#endif
        double n = 0.0, dd = 0.0;
#pragma omp for schedule(static)
        for (int o = 0; o < rows; ++o) {
            const float* prow = dw.W_q_prime.data() + (size_t)o * (4 * d);
            const float* wrow = W_q + (size_t)o * d;
            for (int i = 0; i < d; ++i) {
                double s = 0.0;
                for (int z = 0; z < 4 * d; ++z) {
                    s += (double)prow[z] * dw.W_z[(size_t)z * d + i];
                }
                const double diff = s - (double)wrow[i];
                n += diff * diff;
                dd += (double)wrow[i] * wrow[i];
            }
        }
        partial_num[tid] = n;
        partial_den[tid] = dd;
    }
    double num = 0.0, den = 0.0;
    for (int t = 0; t < max_threads; ++t) { num += partial_num[t]; den += partial_den[t]; }
    if (den <= 0.0) return -1.0;
    return std::sqrt(num / den);
}

} // namespace winnex_xfactor_c
