// latent_coupling.cpp — the Ψ attention mechanism (native C++20).
#include "winnex_xfactor_c/latent_coupling.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace winnex_xfactor_c {

void phase_factors(const float* v, int n_heads, int head_dim,
                   const float* W_phi, int phi_stride,
                   float* re, float* im) {
    const int H = n_heads * head_dim;
    if (!v || !re || !im || H <= 0) return;
    if (!W_phi || phi_stride <= 0) {
        for (int i = 0; i < H; ++i) { re[i] = 1.0f; im[i] = 0.0f; }
        return;
    }
    // phi = W_φ·v  (top-left H×H block, buffer stride = phi_stride)
    for (int i = 0; i < H; ++i) {
        const float* wrow = W_phi + (size_t)i * phi_stride;
        float phi = 0.0f;
        for (int j = 0; j < H; ++j) phi += wrow[j] * v[j];
        re[i] = std::cos(phi);
        im[i] = std::sin(phi);
    }
}

void latent_forward(const float* x, int d, const float* W_z,
                    const float* W_qp, const float* W_rp, const float* W_hp,
                    int n_heads, int n_kv_heads, int head_dim,
                    double phase_alpha, const float* W_phi, int phi_stride,
                    float* Z, float* Q, float* R, float* H) {
    if (!x || !W_z || !W_qp || !W_rp || !W_hp || d <= 0) {
        throw std::runtime_error("latent_forward: invalid input");
    }
    const int d_latent = 4 * d;
    const int d_q = n_heads * head_dim;
    const int d_kv = n_kv_heads * head_dim;

    // ── Step 1: Z = W_z·x ─────────────────────────────────────────────────
    for (int z = 0; z < d_latent; ++z) {
        const float* wzrow = W_z + (size_t)z * d;
        float s = 0.0f;
        for (int i = 0; i < d; ++i) s += wzrow[i] * x[i];
        Z[z] = s;
    }

    // ── Step 2: derived heads (pre-phase, real) ────────────────────────────
    std::vector<float> q_real(d_q), r_real(d_kv);
    auto proj = [&](const float* W, int out, float* dst) {
        for (int o = 0; o < out; ++o) {
            const float* wrow = W + (size_t)o * d_latent;
            float s = 0.0f;
            for (int z = 0; z < d_latent; ++z) s += wrow[z] * Z[z];
            dst[o] = s;
        }
    };
    proj(W_qp, d_q, q_real.data());
    proj(W_rp, d_kv, r_real.data());
    proj(W_hp, d_kv, H);

    // ── Step 3: phase activation Ψ(v) = v ⊙ exp(i·W_φ·v) ──────────────────
    // Q and R are ALWAYS written as complex interleaved pairs (re, im, …).
    //   phase off → Q = [q, 0, q, 0, …]   (score → real dot product)
    //   phase on  → Q = [q·cos φ, q·sin φ, …]
    // The score Re(Q'·R'*) = Σ (Q_re·R_re + Q_im·R_im) = Σ Q·R·cos(φq−φr).
    // H is NOT phase-activated (per the mechanism).
    const int dq2 = 2 * d_q, dkv2 = 2 * d_kv;
    const int stride = (phi_stride > 0) ? phi_stride : d_q;
    if (phase_alpha != 0.0 && W_phi) {
        std::vector<float> re_q(d_q), im_q(d_q), re_r(d_kv), im_r(d_kv);
        phase_factors(q_real.data(), n_heads, head_dim, W_phi, stride, re_q.data(), im_q.data());
        phase_factors(r_real.data(), n_kv_heads, head_dim, W_phi, stride, re_r.data(), im_r.data());
        for (int i = 0; i < d_q; ++i) {
            Q[2*i+0] = q_real[i] * re_q[i];
            Q[2*i+1] = q_real[i] * im_q[i];
        }
        for (int i = 0; i < d_kv; ++i) {
            R[2*i+0] = r_real[i] * re_r[i];
            R[2*i+1] = r_real[i] * im_r[i];
        }
    } else {
        for (int i = 0; i < d_q; ++i) { Q[2*i+0] = q_real[i]; Q[2*i+1] = 0.0f; }
        for (int i = 0; i < d_kv; ++i) { R[2*i+0] = r_real[i]; R[2*i+1] = 0.0f; }
    }
}

float score_phase(const float* Q, int n_heads, int n_kv_heads, int head_dim,
                  int q_head, const float* R_cache) {
    // If phase activation was applied, Q and R_cache are interleaved complex
    // pairs (re, im). The score is Re(Q'·R'*):
    //   Σ_j (Q_re_j·R_re_j + Q_im_j·R_im_j), scaled by 1/√head_dim.
    // We cannot tell from the signature whether phase was applied; the caller
    // passes phase_alpha in the forward. To keep this function correct in both
    // modes we always read the (re, im) layout: with no phase, the caller
    // stores Q as [Q, 0] pairs and the score reduces to the dot product.
    const int kvh = q_head / (n_heads / n_kv_heads);   // GQA mapping
    const float* Qh = Q + (size_t)q_head * (2 * head_dim);
    const float* Rk = R_cache + (size_t)kvh * (2 * head_dim);
    float s = 0.0f;
    for (int j = 0; j < head_dim; ++j) {
        const float qr = Qh[2*j], qi = Qh[2*j+1];
        const float rr = Rk[2*j], ri = Rk[2*j+1];
        s += qr * rr + qi * ri;
    }
    return s / std::sqrt((float)head_dim);
}

} // namespace winnex_xfactor_c
