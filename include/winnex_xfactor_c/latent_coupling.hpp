/**
 * latent_coupling.hpp — the Ψ attention mechanism (native, C++20).
 *
 * Replaces the classical QKV attention block with:
 *
 *   1. Latent coupling:  Z = W_z·x,  Z = LN(Z)          (d_latent = 4·d)
 *   2. Derived heads:    Q = W_q'·Z, R = W_r'·Z, H = W_h'·Z
 *   3. Phase activation: Ψ(v) = v ⊙ exp(i·W_φ·v)        (complex modulation)
 *   4. Similarity:       scores = Re(Q'·R'*),  L2_norm(dim=-1)   (no softmax)
 *   5. Output:           out = scores·H,  out = W_o·out
 *
 * The weights W_z, W_q', W_r', W_h' come from derive_weights (which preserves
 * the classical attention EXACTLY when W_φ = 0). The phase activation is
 * W_φ = α·P where P is the model's X-factor projector (deterministic) — a
 * structural, training-free modulation.
 *
 * All state is real floats; the complex phase is represented as the pair
 * (re, im) and only used where the math requires it (phase factor applied
 * elementwise). This keeps the memory overhead 2× (complex), not more.
 *
 * BSL 1.1 | pay@winnex.ai | (c) Winnex Brasil Soluções Empresariais LTDA-ME
 */
#ifndef WINNEX_XFACTOR_C_LATENT_COUPLING_HPP
#define WINNEX_XFACTOR_C_LATENT_COUPLING_HPP

#include <cstddef>
#include <vector>

#include "winnex_xfactor_c/derive_weights.hpp"

namespace winnex_xfactor_c {

// Options for the Ψ attention mechanism.
struct AttentionConfig {
    int d = 0;          // hidden size
    int n_heads = 0;    // query heads
    int n_kv_heads = 0; // key/value heads (GQA); n_kv_heads | n_heads
    int head_dim = 0;   // d / n_heads
    double phase_alpha = 0.0;  // 0 = identity (classical-equivalent mode)
};

// Computes the phase activation factors: exp(i·W_φ·v) for each head-dim pair.
// phi = W_φ·v  →  (cos(phi), sin(phi))  per element.
// W_φ is a square matrix whose buffer has `phi_stride` columns (the row stride
// in the allocated buffer). The active H×H block is the top-left corner
// (i.e. element (i,j) of the block lives at W_phi[i*phi_stride + j]). This
// lets a single [d_q × d_q] W_phi serve both the query side (stride d_q, block
// d_q×d_q) and the kv side (stride d_q, block d_kv×d_kv). If W_phi is empty,
// returns identity (re=1, im=0).
void phase_factors(const float* v, int n_heads, int head_dim,
                   const float* W_phi,          // square buffer, or nullptr
                   int phi_stride,              // column stride of W_phi
                   float* re, float* im);       // [n_heads·head_dim]

// Latent coupling forward for a SINGLE query position against a KV cache.
//
//   x        : [d]                        the input hidden state at this position
//   Z        : [4d]                       output latent (shared coupling)
//   Q        : [2·n_heads·head_dim]       derived query, COMPLEX interleaved
//   R        : [2·n_kv_heads·head_dim]    derived relational, COMPLEX interleaved
//   H        : [n_kv_heads·head_dim]      derived value (REAL, no phase)
//
// Q and R are ALWAYS written as complex interleaved pairs (re, im, re, im, …):
//   phase off  → Q = [q, 0, q, 0, …]   (the score reduces to a real dot product)
//   phase on   → Q = [q·cos φ, q·sin φ, …]
// This makes the buffer contract unambiguous: callers allocate 2× for Q/R.
//
// Weights (row-major):
//   W_z      : [4d × d]
//   W_q'     : [n_heads·head_dim × 4d]
//   W_r'     : [n_kv_heads·head_dim × 4d]
//   W_h'     : [n_kv_heads·head_dim × 4d]
//
// If phase_alpha != 0 and W_phi != null, the phase activation
//   Ψ(v) = v ⊙ exp(i·W_φ·v)
// is applied to Q and R. W_phi is a square buffer (row-major) with
// `phi_stride` columns; the active block is the top-left corner of the side
// being activated:
//   query side → block [d_q × d_q] starting at offset 0, stride phi_stride
//   kv side    → block [d_kv × d_kv] starting at offset 0, stride phi_stride
// So a single [d_q × d_q] W_phi works for both when phi_stride = d_q.
void latent_forward(const float* x, int d, const float* W_z,
                    const float* W_qp, const float* W_rp, const float* W_hp,
                    int n_heads, int n_kv_heads, int head_dim,
                    double phase_alpha, const float* W_phi, int phi_stride,
                    float* Z, float* Q, float* R, float* H);

// Attention scores between the query position's Q and a cached key position:
//   score = Re(Q'·R'*)
// Q and R are the phase-activated heads [n_heads·head_dim]; the KV cache
// stores R (the phase-activated "key") for each position. With GQA, each
// query head hh attends to kv-head (hh / (n_heads/n_kv_heads)).
// Returns the real score (scaled by 1/sqrt(head_dim), as classical).
float score_phase(const float* Q, int n_heads, int n_kv_heads, int head_dim,
                  int q_head, const float* R_cache);

} // namespace winnex_xfactor_c

#endif // WINNEX_XFACTOR_C_LATENT_COUPLING_HPP
