/**
 * psi_forward.hpp — Ψ forward engine: the latent-coupling attention integrated
 * into the full transformer forward (24 layers), replacing the classical
 * QKV+RoPE+attention block. Uses the REAL model weights (adapted via the
 * pseudoinverse, exact when W_φ=0) plus the additive positional phase and the
 * spectral log-phase filter (AC). This is the experiment to answer:
 *   "does Ψ improve quality? is it cheaper?"
 *
 * Per layer, the Ψ block replaces QKV+RoPE+attention:
 *   1. RMSNorm (reuses winnex-nano rms_norm)
 *   2. Latent coupling (GPU): Z = W_z·x → Q/R/H, + bias
 *   3. Additive positional phase: Q/R rotated by pos·π/128 (replaces RoPE)
 *   4. [AC] spectral log-phase filter: F(k) on Q
 *   5. Madhava top-k attention: Re(Q'·R'*) + top-16 + weight=score/Σ
 *   6. o_proj (reused from the model)
 *   7. residual → post_norm → MLP (gate/up/down, SiLU) → residual
 * Final: norm + lm_head → logits.
 *
 * The weights W_z/W_q'/W_r'/W_h' are DERIVED from the model's QKV (exact
 * reparametrization when W_φ=0 and no positional/spectral modulation).
 *
 * BSL 1.1 | pay@winnex.ai | (c) Winnex Brasil Soluções Empresariais LTDA-ME
 */
#ifndef WINNEX_XFACTOR_C_PSI_FORWARD_HPP
#define WINNEX_XFACTOR_C_PSI_FORWARD_HPP

#include <cstddef>
#include <string>
#include <vector>

namespace winnex_xfactor_c {

// Config for the Ψ forward (mirrors ModelConfig).
struct PsiModelConfig {
    std::string arch = "qwen2";
    int hidden_size = 0;
    int intermediate_size = 0;
    int num_hidden_layers = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;
    int vocab_size = 0;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;   // RoPE base (Qwen2.5 uses 1e6)
    bool tie_word_embeddings = true;
};

// The derived latent weights for ONE layer + the biases.
struct PsiLayerWeights {
    std::vector<float> W_z;        // [4d × d]
    std::vector<float> W_qp;       // [d_q × 4d]
    std::vector<float> W_rp;       // [d_kv × 4d]
    std::vector<float> W_hp;       // [d_kv × 4d]
    std::vector<float> b_q;        // [d_q]
    std::vector<float> b_k;        // [d_kv]
    std::vector<float> b_v;        // [d_kv]
    std::vector<float> in_norm;    // [d]
    std::vector<float> post_norm;  // [d]
    std::vector<float> o_proj;     // [d × d_q]
    std::vector<float> g_w, u_w, d_w;  // MLP [inter×d], [inter×d], [d×inter]
};

// Ψ forward engine: runs the full model with the latent-coupling attention.
class PsiForwardEngine {
public:
    // Loads the model and derives the latent weights per layer (cacheable).
    //   model_dir : dir with config.json + model.safetensors
    //   cache_path: optional path to save/load the derived weights (.npz)
    PsiForwardEngine(const std::string& model_dir, const std::string& cache_path = "");

    // Runs the forward on a sequence of hidden states [seq, d].
    // Returns logits [seq, vocab] (all positions) or [vocab] (last).
    std::vector<float> forward(const std::vector<float>& hidden, int seq_len,
                               bool all_positions = false);

    // Autoregressive next-token (persistent KV cache).
    std::vector<float> forward_next(const std::vector<float>& hidden);

    // Modes.
    void set_spectral_alpha(double alpha);   // AC filter strength (0 = off)
    void set_phase_alpha(double alpha);      // W_φ activation (0 = off)
    void set_positional(bool on);            // additive positional phase

    // Diagnostics.
    int num_layers() const { return cfg_.num_hidden_layers; }
    int hidden_size() const { return cfg_.hidden_size; }
    int vocab_size() const { return cfg_.vocab_size; }

    // Saves/loads the derived weights cache.
    void save_cache(const std::string& path) const;
    bool load_cache(const std::string& path);

private:
    PsiModelConfig cfg_;
    std::vector<PsiLayerWeights> layers_;
    std::vector<float> final_norm_;
    std::vector<float> embed_tokens_;   // [vocab, d] (tied lm_head)
    double spectral_alpha_ = 0.0;
    bool positional_ = true;

    // Persistent per-layer caches (R complex + H real).
    std::vector<std::vector<float>> r_caches_, h_caches_;
    int ctx_len_ = 0;

    void run_layer(int layer, std::vector<float>& h, int seq_len, int cache_pos);
    double phase_alpha_ = 0.0;
    std::vector<float> phase_P_;
};

} // namespace winnex_xfactor_c

#endif // WINNEX_XFACTOR_C_PSI_FORWARD_HPP
