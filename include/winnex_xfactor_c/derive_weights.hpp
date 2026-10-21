/**
 * derive_weights.hpp — adaptive weight derivation: QKV → latent coupling.
 *
 * Given a pretrained dense model's Q/K/V projection weights, derive a latent
 * coupling parametrization that PRESERVES the classical attention EXACTLY
 * (when the phase activation is off). This is the "adaptability" of the
 * winnex-xfactor-c engine: no training, deterministic, drop-in.
 *
 * ──────────────────────────────────────────────────────────────────────────
 *  The mathematics
 * ──────────────────────────────────────────────────────────────────────────
 *  Classical (row-major [out, in]):
 *      Q = W_q·x,  K = W_k·x,  V = W_v·x
 *
 *  Latent coupling (d_latent = 4·d):
 *      Z   = W_z·x,          W_z ∈ ℝ^{4d×d},  Z ∈ ℝ^{4d}
 *      Q'  = W_q'·Z          (derived)
 *      R'  = W_r'·Z          (derived)
 *      H'  = W_h'·Z          (derived)
 *
 *  Choice of W_z (structural, orthonormal blocks):
 *      W_z = [ I ; R₁ ; R₂ ; R₃ ] ∈ ℝ^{4d×d}
 *      with Rᵢ unit quaternion rotations → RᵢᵀRᵢ = I.
 *      Then  W_zᵀ·W_z = I + R₁ᵀR₁ + R₂ᵀR₂ + R₃ᵀR₃ = 4·I_d
 *      and   W_z⁺ = (1/4)·W_zᵀ  is the left-inverse (Moore–Penrose):
 *            W_z⁺·W_z = (1/4)·W_zᵀ·W_z = I_d.
 *
 *  Derivation:
 *      W_q' = W_q·W_z⁺ = (1/4)·W_q·W_zᵀ   ∈ ℝ^{d_q×4d}
 *      W_r' = W_k·W_z⁺ = (1/4)·W_k·W_zᵀ   ∈ ℝ^{d_kv×4d}
 *      W_h' = W_v·W_z⁺ = (1/4)·W_v·W_zᵀ   ∈ ℝ^{d_kv×4d}
 *
 *  Preservation (exact):
 *      Q' = W_q'·Z = (1/4)·W_q·W_zᵀ·W_z·x = (1/4)·W_q·(4·I_d)·x = W_q·x
 *
 *  So with W_φ = 0 (no phase activation), the latent-coupling attention
 *  computes IDENTICAL Q/R/H to the classical QKV — a reparametrization.
 *
 *  GQA: W_z is shared; W_q [d_q×d], W_k/W_v [d_kv×d] derive independently.
 *
 * BSL 1.1 | pay@winnex.ai | (c) Winnex Brasil Soluções Empresariais LTDA-ME
 */
#ifndef WINNEX_XFACTOR_C_DERIVE_WEIGHTS_HPP
#define WINNEX_XFACTOR_C_DERIVE_WEIGHTS_HPP

#include <cstddef>
#include <vector>

namespace winnex_xfactor_c {

// Derives the latent-coupling weights from the classical Q/K/V projection
// weights (row-major [out, in]).
//
// Params:
//   W_q, W_k, W_v : row-major projection weights.
//                   W_q: [d_q × d], W_k: [d_kv × d], W_v: [d_kv × d]
//   d             : input embedding dimension (hidden_size).
//   seed          : deterministic seed for the rotation blocks R₁..R₃.
//
// Returns (W_z, W_q_prime, W_r_prime, W_h_prime):
//   W_z       : [4d × d]  structural latent projection (orthonormal blocks).
//   W_q_prime : [d_q × 4d]  derived query projection.
//   W_r_prime : [d_kv × 4d] derived relational (key) projection.
//   W_h_prime : [d_kv × 4d] derived value projection.
struct DerivedWeights {
    std::vector<float> W_z;        // [4d × d]  (row-major)
    std::vector<float> W_q_prime;  // [d_q × 4d] (row-major)
    std::vector<float> W_r_prime;  // [d_kv × 4d] (row-major)
    std::vector<float> W_h_prime;  // [d_kv × 4d] (row-major)

    int d = 0;       // input dimension
    int d_q = 0;     // query output dim
    int d_kv = 0;    // key/value output dim
};

DerivedWeights derive_weights(const float* W_q, const float* W_k,
                              const float* W_v, int d, int d_q, int d_kv,
                              unsigned seed = 42);

// Verifies the preservation identity ‖W_q'·W_z − W_q‖_F (relative).
// W_q is the ORIGINAL query weight [d_q × d]. Returns the relative
// Frobenius error (0 == exact reparametrization). max_rows samples the
// first min(max_rows, d_q) output rows (cheaper for large models; the
// pseudoinverse is uniform in rows, so a sample is representative).
double preservation_error(const DerivedWeights& dw, const float* W_q,
                          int max_rows = 0);  // 0 = all rows

} // namespace winnex_xfactor_c

#endif // WINNEX_XFACTOR_C_DERIVE_WEIGHTS_HPP
