/**
 * attention_forward.hpp — the Ψ attention mechanism, OpenCL-optimized.
 *
 * HilbertAttention: the Ψ attention forward with a KV cache optimized in the
 * complex Hilbert space and executed on a generic GPU via runtime-loaded
 * OpenCL (dlopen + JIT kernels, the winnex-madhava pattern).
 *
 * HILBERT-SPACE OPTIMIZATION
 * --------------------------
 * The mechanism's state is a complex wavefunction:
 *   Q' = Q ⊙ e^{i·φq},   R' = R ⊙ e^{i·φr}   (φ = W_φ·v)
 * The score is the Hilbert inner product (real part of the Hermitian one):
 *   score(q,k) = Re(⟨Q'_q, R'_k⟩) = Σ_j (Q_re·R_re + Q_im·R_im)
 * Two optimizations follow from working in Hilbert space directly:
 *   1. The phase is applied ONCE at latent time (latent_forward); the complex
 *      key R' is cached as the (re,im) pair. Attention NEVER recomputes
 *      cos/sin — the cached complex key IS the phase-activated state.
 *   2. The score kernel is a FUSED complex inner product over the interleaved
 *      (re,im) pairs — no decomposition to real vectors, no re-im layout
 *      shuffle. The (re,im) layout is exactly the device cache layout.
 *
 * KV CACHE (OpenCL)
 * -----------------
 * The R/H caches are allocated ON the device and grown by doubling. An
 * autoregressive decode appends one position's R/H to device memory and
 * attends without host round-trip of the context.
 *
 * OpenCL is REQUIRED — there is NO CPU fallback. The constructor throws if no
 * OpenCL loader/device is present.
 *
 * BSL 1.1 | pay@winnex.ai | (c) Winnex Brasil Soluções Empresariais LTDA-ME
 */
#ifndef WINNEX_XFACTOR_C_ATTENTION_FORWARD_HPP
#define WINNEX_XFACTOR_C_ATTENTION_FORWARD_HPP

#include <cstddef>
#include <vector>

namespace winnex_xfactor_c {

// HilbertAttention — Ψ attention forward with an OpenCL-optimized Hilbert-space
// KV cache. One instance per layer (owns that layer's device cache + kernels).
class HilbertAttention {
public:
    // Builds the engine for a GQA geometry.
    //   d_q, d_kv : query/kv output dims (d_q = n_heads·head_dim, d_kv = n_kv_heads·head_dim)
    //   d         : model hidden size (o_proj output dim)
    // Throws std::runtime_error if OpenCL is unavailable (no fallback).
    HilbertAttention(int d_q, int d_kv, int n_heads, int n_kv_heads,
                     int head_dim, int d);
    ~HilbertAttention();

    HilbertAttention(const HilbertAttention&) = delete;
    HilbertAttention& operator=(const HilbertAttention&) = delete;

    // Uploads the output projection (row-major [d × d_q]) once.
    // y = o_proj · (scores·H). Optional: without it, attention() fills only out.
    void set_output_projection(const float* o_proj);

    // Attention biases (Qwen2/Qwen2.5 carry q/k/v biases; Llama does not).
    // Pass nullptr / empty to leave them at zero.
    void set_attention_bias(const float* b_q, const float* b_k, const float* b_v);

    // RoPE (rotate_half) applied to Q/R inside the QRH kernel. `theta` is the
    // model's rope_theta. Enabled by default: a pretrained model cannot be run
    // without its positional encoding.
    void set_rope(float theta, bool enabled = true);

    // Appends one position's phase-activated key R and value H to the DEVICE
    // cache (the Hilbert-space complex key; phase already applied).
    //   R : [2·d_kv] complex interleaved (re,im) pairs  (from latent_forward)
    //   H : [d_kv]   real value                         (from latent_forward)
    void kv_append(const float* R, const float* H);

    // GPU LATENT FORWARD (the operational correlation): runs the entire latent
    // coupling on the device in ONE fused kernel —
    //   Z   = W_z·x            (d_latent = 4·d)
    //   Q/R/H = W_q'/r'/h'·Z
    // with phase activation applied on-device (W_φ = α·W_phi or none).
    // This is the GPU-accelerated replacement for the CPU `latent_forward`
    // (which was 98.7% of the pipeline). The weights must be uploaded once via
    // set_latent_weights. On success, Q_psi is written to `out` [2·d_q]
    // (complex interleaved) and R_psi to `out_r` [2·d_kv]; both can then be
    // appended to the cache directly (kv_append from device, see below).
    //   x   : [d] input hidden state (host)
    //   out : [2·d_q] phase-activated query (host)
    //   out_r: [2·d_kv] phase-activated relational/key (host)
    //   phase_alpha : 0 = no phase, else α for the W_φ modulation.
    void latent_forward_gpu(const float* x, float* out, float* out_r,
                            double phase_alpha = 0.0);

    // Uploads the latent-coupling weights (once per layer):
    //   W_z  : [4d × d]   structural latent projection
    //   W_qp : [d_q × 4d] derived query
    //   W_rp : [d_kv × 4d] derived relational
    //   W_hp : [d_kv × 4d] derived value
    void set_latent_weights(const float* W_z, const float* W_qp,
                            const float* W_rp, const float* W_hp);

    // Appends one position's R/H to the cache DIRECTLY from device buffers
    // (no host round-trip). R/H are the device outputs of latent_forward_gpu.
    void kv_append_gpu();

    // FULL GPU PIPELINE for one token (no host round-trip of Q/R):
    //   latent (Z → Q/R/H, on device, appended DIRECTLY to the cache) →
    //   fused attention (scores+L2+apply, one kernel) → o_proj.
    // Everything stays on the device; only `y` [d] is read back. This is the
    // fast autoregressive path: ONE upload (x) + 4 kernels + ONE blocking read.
    //   x    : [d] input hidden state
    //   y    : [d] attention output (o_proj applied); read back to host.
    //   phase_alpha : phase modulation strength (0 = off).
    void forward_gpu(const float* x, float* y, double phase_alpha = 0.0);


    // Uploads the phase matrix W_φ [d_q × d_q] (row-major). The phase
    // activation Ψ(v)=v⊙exp(i·W_φ·v) is then applied on-device when
    // latent_forward_gpu is called with phase_alpha != 0. The kv side reads
    // the top-left d_kv × d_kv block (stride d_q).
    void set_phase_matrix(const float* W_phi);

    // Spectral log-phase filter: applies F(k)=exp(i·α·arctan(ln(k+ε))) to the
    // phase-activated Q before the attention (the harmonic modulation). α=0
    // disables it. This is a deterministic spectral regularizer (compresses
    // high-frequency phase components), not a trained layer.
    void set_spectral_alpha(double alpha);

    // Applies the spectral log-phase filter to a complex-interleaved vector in
    // place (the SAME kernel used by forward_gpu). Exposed for isolated testing
    // of the spectral component (C6). v is [2·n] (re,im) pairs; alpha is the
    // modulation strength.
    void apply_spectral(float* v, int n, double alpha);

    // Full Ψ attention for ONE query position against the device cache.
    //   Q   : [2·d_q] phase-activated query (complex interleaved), from latent_forward
    //   out : [d_q]   concatenated per-head values = scores·H (L2 normalized)
    //   y   : [d]     after o_proj (only if set_output_projection was called)
    void attention(const float* Q, float* out, float* y);

    // Current context length in the device cache.
    int ctx_len() const;
    // Clears the device cache (start a new generation).
    void reset();

    // Geometry (for bindings/diagnostics).
    int d_q() const;
    int d_kv() const;
    int d() const;
    // Device name (for diagnostics).
    const char* device_name() const;

private:
    struct Impl;
    Impl* impl_;
};

} // namespace winnex_xfactor_c

#endif // WINNEX_XFACTOR_C_ATTENTION_FORWARD_HPP
