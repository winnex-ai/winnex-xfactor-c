// test_attention_forward.cpp — validates the OpenCL Hilbert attention.
//
// Gate: the OpenCL attention output must match a reference CPU computation of
//   scores = Re(Q'·R'*) / √head_dim, Madhava selective top-k (ksel=16),
//   weight = score/Σ(top-k), out[h][j] = Σ_k scores[k]·H[k][kvh][j]
// within float32 tolerance. This validates the device kernels against the math.
#include "winnex_xfactor_c/attention_forward.hpp"
#include "winnex_xfactor_c/latent_coupling.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include <cmath>

using winnex_xfactor_c::HilbertAttention;
using winnex_xfactor_c::latent_forward;

// Reference CPU implementation of the attention (for validation only).
static void ref_attention(const std::vector<float>& Q,     // [2·d_q]
                          const std::vector<float>& Rcache,// [ctx·2·d_kv]
                          const std::vector<float>& Hcache,// [ctx·d_kv]
                          int n_heads, int n_kv, int hd,
                          std::vector<float>& out) {       // [n_heads·hd]
    const int d_q = n_heads * hd, d_kv = n_kv * hd;
    const int ctx = (int)Hcache.size() / d_kv;
    out.assign(d_q, 0.0f);
    for (int h = 0; h < n_heads; ++h) {
        const int kvh = h / (n_heads / n_kv);
        // scores over ctx
        std::vector<float> scores(ctx);
        for (int k = 0; k < ctx; ++k) {
            float s = 0.0f;
            for (int j = 0; j < hd; ++j) {
                float qr = Q[2*(h*hd+j)], qi = Q[2*(h*hd+j)+1];
                float rr = Rcache[(size_t)(k*n_kv+kvh)*2*hd + 2*j];
                float ri = Rcache[(size_t)(k*n_kv+kvh)*2*hd + 2*j + 1];
                s += qr*rr + qi*ri;
            }
            scores[k] = s / std::sqrt((float)hd);
        }
        // REFERENCE UPDATED (2026-10-21): standard causal SOFTMAX over every
        // visible key — the function the pretrained weights encode.
        //
        // The previous reference implemented the kernel's old behaviour
        // (top-16, weight = score/Σscore, no softmax). Attention scores are
        // SIGNED, so that weighting is unbounded and Σ can pass through zero;
        // on Qwen2.5-0.5B it produced perplexity 8.4e6 against 35.1 for
        // transformers. Truncating to 16 keys also silently drops context on
        // any sequence longer than 16 tokens.
        //
        // The kernel now computes softmax; this reference must match it, not
        // the removed behaviour.
        float m = scores[0];
        for (int k = 1; k < ctx; ++k) if (scores[k] > m) m = scores[k];
        float wsum = 0.0f;
        for (int k = 0; k < ctx; ++k) { scores[k] = std::exp(scores[k] - m); wsum += scores[k]; }
        const float inv = (wsum > 0.0f) ? 1.0f / wsum : 0.0f;
        // out[h][j] = Σ_k softmax_k · H[k][kvh][j]
        for (int j = 0; j < hd; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < ctx; ++k)
                acc += (scores[k] * inv) * Hcache[(size_t)(k*n_kv+kvh)*hd + j];
            out[h*hd + j] = acc;
        }
    }
}

int main() {
    int failures = 0;

    const int n_heads = 4, n_kv = 2, hd = 8;
    const int d_q = n_heads * hd, d_kv = n_kv * hd, d = 16;
    const int ctx = 5;

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // Build a HilbertAttention (requires OpenCL — the test FAILS if absent).
    HilbertAttention attn(d_q, d_kv, n_heads, n_kv, hd, d);
    printf("device: %s\n", attn.device_name());

    // Random query (complex interleaved) and cache.
    std::vector<float> Q(2 * d_q);
    for (auto& v : Q) v = nd(rng);
    std::vector<float> Rcache(ctx * 2 * d_kv), Hcache(ctx * d_kv);
    for (auto& v : Rcache) v = nd(rng);
    for (auto& v : Hcache) v = nd(rng);

    // Push the cache in two appends to exercise growth (cap doubling).
    for (int k = 0; k < 3; ++k) {
        attn.kv_append(Rcache.data() + (size_t)k * 2 * d_kv,
                       Hcache.data() + (size_t)k * d_kv);
    }
    for (int k = 3; k < ctx; ++k) {
        attn.kv_append(Rcache.data() + (size_t)k * 2 * d_kv,
                       Hcache.data() + (size_t)k * d_kv);
    }
    if (attn.ctx_len() != ctx) {
        printf("ctx_len=%d (want %d) [FAIL]\n", attn.ctx_len(), ctx);
        ++failures;
    }

    // Run the OpenCL attention.
    std::vector<float> out(d_q), y(d);
    attn.attention(Q.data(), out.data(), y.data());

    // Reference CPU.
    std::vector<float> ref(d_q);
    ref_attention(Q, Rcache, Hcache, n_heads, n_kv, hd, ref);

    // Compare.
    double max_err = 0.0;
    for (int i = 0; i < d_q; ++i) {
        double e = std::fabs((double)out[i] - ref[i]);
        if (e > max_err) max_err = e;
    }
    bool ok = (max_err < 1e-3);
    printf("attention max|out − ref| = %.3e [%s]\n", max_err, ok ? "OK" : "FAIL");
    if (!ok) {
        for (int i = 0; i < d_q; ++i)
            printf("  out[%d]=%.6f ref[%d]=%.6f\n", i, out[i], i, ref[i]);
        ++failures;
    }

    // ── GPU latent forward: compare Q/R/H against the CPU latent_forward ──
    {
        const int dd = 32;              // hidden
        const int dd_q = n_heads * hd;  // 32
        const int dd_kv = n_kv * hd;    // 16
        // Random QKV → derive → same weights both sides.
        std::vector<float> Wq(dd_q * dd), Wk(dd_kv * dd), Wv(dd_kv * dd);
        for (auto& v : Wq) v = nd(rng);
        for (auto& v : Wk) v = nd(rng);
        for (auto& v : Wv) v = nd(rng);
        auto dw = winnex_xfactor_c::derive_weights(Wq.data(), Wk.data(), Wv.data(),
                                                   dd, dd_q, dd_kv, 11);

        HilbertAttention attn2(dd_q, dd_kv, n_heads, n_kv, hd, dd);
        attn2.set_latent_weights(dw.W_z.data(), dw.W_q_prime.data(),
                                 dw.W_r_prime.data(), dw.W_h_prime.data());

        std::vector<float> x(dd);
        for (auto& v : x) v = nd(rng);

        // CPU latent_forward reference.
        std::vector<float> Zc(4 * dd), Qc(2 * dd_q), Rc(2 * dd_kv), Hc(dd_kv);
        latent_forward(x.data(), dd, dw.W_z.data(), dw.W_q_prime.data(),
                       dw.W_r_prime.data(), dw.W_h_prime.data(),
                       n_heads, n_kv, hd, 0.0, nullptr, 0,
                       Zc.data(), Qc.data(), Rc.data(), Hc.data());

        // GPU latent_forward.
        std::vector<float> Qg(2 * dd_q), Rg(2 * dd_kv);
        attn2.latent_forward_gpu(x.data(), Qg.data(), Rg.data(), 0.0);

        double maxe = 0.0;
        for (int i = 0; i < 2 * dd_q; ++i) {
            double e = std::fabs((double)Qg[i] - Qc[i]);
            if (e > maxe) maxe = e;
        }
        for (int i = 0; i < 2 * dd_kv; ++i) {
            double e = std::fabs((double)Rg[i] - Rc[i]);
            if (e > maxe) maxe = e;
        }
        bool gok = (maxe < 1e-3);
        printf("GPU latent Q/R vs CPU ref: max|Δ| = %.3e [%s]\n",
               maxe, gok ? "OK" : "FAIL");
        if (!gok) ++failures;

        // kv_append_gpu: append the GPU-computed R/H and check ctx_len grows.
        attn2.kv_append_gpu();
        bool cap = (attn2.ctx_len() == 1);
        printf("kv_append_gpu → ctx_len = %d [%s]\n", attn2.ctx_len(), cap ? "OK" : "FAIL");
        if (!cap) ++failures;
    }

    if (failures == 0) {
        printf("\nALL ATTENTION-FORWARD TESTS PASSED\n");
        return 0;
    }
    printf("\n%d FAILURE(S)\n", failures);
    return 1;
}
