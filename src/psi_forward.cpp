// psi_forward.cpp — Ψ forward engine (latent coupling in the full model).
#include "winnex_xfactor_c/psi_forward.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

#include "winnex_nano/safetensors.hpp"
#include "winnex_nano/tensor_ops.hpp"
#include "winnex_nano/x_factor.hpp"
#include "winnex_xfactor_c/attention_forward.hpp"
#include "winnex_xfactor_c/derive_weights.hpp"

namespace winnex_xfactor_c {

// Minimal config.json loader (mirrors winnex-nano load_config but returns our
// PsiModelConfig). Reads top-level int/float/string fields.
namespace {
PsiModelConfig load_psi_config(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("PsiForward: cannot open " + path);
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto find_int = [&](const std::string& key, int def) {
        std::string pat = "\"" + key + "\"";
        size_t p = s.find(pat);
        if (p == std::string::npos) return def;
        p = s.find(':', p); if (p == std::string::npos) return def;
        p++;
        while (p < s.size() && (s[p]==' '||s[p]=='\n'||s[p]=='\t')) p++;
        int v = 0; bool neg = false;
        if (p < s.size() && s[p]=='-') { neg = true; p++; }
        while (p < s.size() && s[p]>='0' && s[p]<='9') { v = v*10 + (s[p]-'0'); p++; }
        return neg ? -v : v;
    };
    auto find_float = [&](const std::string& key, float def) {
        std::string pat = "\"" + key + "\"";
        size_t p = s.find(pat);
        if (p == std::string::npos) return def;
        p = s.find(':', p); if (p == std::string::npos) return def;
        p++;
        while (p < s.size() && (s[p]==' '||s[p]=='\n'||s[p]=='\t')) p++;
        return (float)std::atof(s.c_str() + p);
    };
    PsiModelConfig c;
    c.arch = "qwen2";
    c.hidden_size = find_int("hidden_size", 0);
    c.intermediate_size = find_int("intermediate_size", 0);
    c.num_hidden_layers = find_int("num_hidden_layers", 0);
    c.num_attention_heads = find_int("num_attention_heads", 0);
    c.num_key_value_heads = find_int("num_key_value_heads", c.num_attention_heads);
    c.vocab_size = find_int("vocab_size", 0);
    c.rms_norm_eps = find_float("rms_norm_eps", 1e-6f);
    c.rope_theta = find_float("rope_theta", 10000.0f);
    c.tie_word_embeddings = find_int("tie_word_embeddings", 1) == 1;
    c.head_dim = c.hidden_size / c.num_attention_heads;
    if (c.hidden_size <= 0 || c.num_hidden_layers <= 0)
        throw std::runtime_error("PsiForward: invalid config");
    return c;
}
} // namespace

// ──────────────────────────────────────────────────────────────────────────
// Constructor: load the model + derive the latent weights per layer.
// ──────────────────────────────────────────────────────────────────────────
PsiForwardEngine::PsiForwardEngine(const std::string& model_dir,
                                   const std::string& cache_path) {
    cfg_ = load_psi_config(model_dir + "/config.json");
    winnex_nano::Safetensors st(model_dir + "/model.safetensors");

    final_norm_ = st.to_float(st.get("model.norm.weight"));

    // X-Factor projector P of the embedding manifold (D x D), used as the
    // phase generator: W_phi = alpha * P. Built once from a sample of
    // embed_tokens via power iteration (winnex_nano::XFactor).
    {
        const int Dm = cfg_.hidden_size;
        const int Vm = cfg_.vocab_size;
        std::vector<float> Esamp((size_t)std::min(Vm, 20000) * Dm);
        const int ns = std::min(Vm, 20000);
        const std::vector<float> Eall = st.to_float(st.get("model.embed_tokens.weight"));
        for (int i = 0; i < ns; ++i)
            for (int j = 0; j < Dm; ++j)
                Esamp[(size_t)i * Dm + j] = Eall[(size_t)i * Dm + j];
        try {
            winnex_nano::XFactor xf(Esamp.data(), ns, Dm, 0.95);
            phase_P_.resize((size_t)Dm * Dm);
            std::vector<float> row(Dm);
            for (int a = 0; a < Dm; ++a) {
                std::vector<float> e((size_t)Dm, 0.0f); e[(size_t)a] = 1.0f;
                xf.project(e.data(), row.data());          // P·e_a = column a
                for (int b = 0; b < Dm; ++b)
                    phase_P_[(size_t)b * Dm + a] = row[(size_t)b];
            }
        } catch (...) {
            phase_P_.clear();   // no projector → phase stays disabled
        }
    }
    if (st.has("lm_head.weight")) {
        embed_tokens_ = st.to_float(st.get("lm_head.weight"));
    } else {
        embed_tokens_ = st.to_float(st.get("model.embed_tokens.weight"));
    }

    if (!cache_path.empty() && load_cache(cache_path)) {
        // Cache loaded the latent weights; reload norms/MLP from the model.
        const int D = cfg_.hidden_size;
        for (int l = 0; l < cfg_.num_hidden_layers; ++l) {
            std::string L = "model.layers." + std::to_string(l) + ".";
            auto& W = layers_[l];
            W.in_norm = st.to_float(st.get(L + "input_layernorm.weight"));
            W.post_norm = st.to_float(st.get(L + "post_attention_layernorm.weight"));
            W.o_proj = st.to_float(st.get(L + "self_attn.o_proj.weight"));
            W.g_w = st.to_float(st.get(L + "mlp.gate_proj.weight"));
            W.u_w = st.to_float(st.get(L + "mlp.up_proj.weight"));
            W.d_w = st.to_float(st.get(L + "mlp.down_proj.weight"));
        }
        return;
    }

    const int D = cfg_.hidden_size;
    const int D_Q = cfg_.num_attention_heads * cfg_.head_dim;
    const int D_KV = cfg_.num_key_value_heads * cfg_.head_dim;
    layers_.resize(cfg_.num_hidden_layers);

    for (int l = 0; l < cfg_.num_hidden_layers; ++l) {
        std::string L = "model.layers." + std::to_string(l) + ".";
        auto& W = layers_[l];
        W.in_norm = st.to_float(st.get(L + "input_layernorm.weight"));
        W.post_norm = st.to_float(st.get(L + "post_attention_layernorm.weight"));

        auto Wq = st.to_float(st.get(L + "self_attn.q_proj.weight"));
        auto Wk = st.to_float(st.get(L + "self_attn.k_proj.weight"));
        auto Wv = st.to_float(st.get(L + "self_attn.v_proj.weight"));

        if (st.has(L + "self_attn.q_proj.bias")) W.b_q = st.to_float(st.get(L + "self_attn.q_proj.bias"));
        if (st.has(L + "self_attn.k_proj.bias")) W.b_k = st.to_float(st.get(L + "self_attn.k_proj.bias"));
        if (st.has(L + "self_attn.v_proj.bias")) W.b_v = st.to_float(st.get(L + "self_attn.v_proj.bias"));

        DerivedWeights dw = derive_weights(Wq.data(), Wk.data(), Wv.data(),
                                           D, D_Q, D_KV, (unsigned)(100 + l));
        W.W_z = std::move(dw.W_z);
        W.W_qp = std::move(dw.W_q_prime);
        W.W_rp = std::move(dw.W_r_prime);
        W.W_hp = std::move(dw.W_h_prime);

        W.o_proj = st.to_float(st.get(L + "self_attn.o_proj.weight"));
        W.g_w = st.to_float(st.get(L + "mlp.gate_proj.weight"));
        W.u_w = st.to_float(st.get(L + "mlp.up_proj.weight"));
        W.d_w = st.to_float(st.get(L + "mlp.down_proj.weight"));
    }

    if (!cache_path.empty()) save_cache(cache_path);
}

// ──────────────────────────────────────────────────────────────────────────
// Cache save/load.
// ──────────────────────────────────────────────────────────────────────────
namespace {
void write_vec(std::ofstream& f, const std::vector<float>& v) {
    size_t n = v.size();
    f.write((const char*)&n, sizeof(size_t));
    f.write((const char*)v.data(), n * sizeof(float));
}
bool read_vec(std::ifstream& f, std::vector<float>& v) {
    size_t n = 0;
    f.read((char*)&n, sizeof(size_t));
    if (!f || n > 1ull << 32) return false;
    v.resize(n);
    f.read((char*)v.data(), n * sizeof(float));
    return (bool)f;
}
} // namespace

void PsiForwardEngine::save_cache(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) return;
    int nl = (int)layers_.size();
    f.write((const char*)&nl, sizeof(int));
    for (const auto& W : layers_) {
        write_vec(f, W.W_z); write_vec(f, W.W_qp); write_vec(f, W.W_rp); write_vec(f, W.W_hp);
        write_vec(f, W.b_q); write_vec(f, W.b_k); write_vec(f, W.b_v);
    }
}

bool PsiForwardEngine::load_cache(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    int nl = 0;
    f.read((char*)&nl, sizeof(int));
    if (!f || nl != cfg_.num_hidden_layers) return false;
    layers_.resize(nl);
    for (auto& W : layers_) {
        if (!read_vec(f, W.W_z) || !read_vec(f, W.W_qp) || !read_vec(f, W.W_rp) ||
            !read_vec(f, W.W_hp) || !read_vec(f, W.b_q) || !read_vec(f, W.b_k) ||
            !read_vec(f, W.b_v)) return false;
    }
    return true;
}

// ──────────────────────────────────────────────────────────────────────────
// Modes.
// ──────────────────────────────────────────────────────────────────────────
void PsiForwardEngine::set_spectral_alpha(double a) { spectral_alpha_ = a; }
void PsiForwardEngine::set_phase_alpha(double a) { phase_alpha_ = a; }
void PsiForwardEngine::set_positional(bool on) { positional_ = on; }

// ──────────────────────────────────────────────────────────────────────────
// forward: process the sequence through all layers.
//
// The Ψ attention per layer uses HilbertAttention on the GPU. For correctness
// (the benchmark measures QUALITY, not speed), each layer's attention keeps a
// persistent HilbertAttention across the sequence: we build one per layer here
// and feed tokens sequentially so the KV cache accumulates the context.
// ──────────────────────────────────────────────────────────────────────────
std::vector<float> PsiForwardEngine::forward(const std::vector<float>& hidden,
                                             int seq_len, bool all_positions) {
    const int D = cfg_.hidden_size;
    if ((int)hidden.size() != seq_len * D)
        throw std::runtime_error("PsiForward: hidden size mismatch");
    const int D_Q = cfg_.num_attention_heads * cfg_.head_dim;
    const int D_KV = cfg_.num_key_value_heads * cfg_.head_dim;

    // Per-layer HilbertAttention (GPU), persistent across the sequence.
    std::vector<HilbertAttention*> attns(cfg_.num_hidden_layers, nullptr);
    for (int l = 0; l < cfg_.num_hidden_layers; ++l) {
        attns[l] = new HilbertAttention(D_Q, D_KV, cfg_.num_attention_heads,
                                        cfg_.num_key_value_heads, cfg_.head_dim, D);
        const auto& W = layers_[l];
        attns[l]->set_latent_weights(W.W_z.data(), W.W_qp.data(), W.W_rp.data(), W.W_hp.data());
        attns[l]->set_output_projection(W.o_proj.data());
        // Qwen2/Qwen2.5 carry q/k/v attention biases; the Ψ path must apply
        // them or every projection is wrong (measured: perplexity 8.4e6).
        attns[l]->set_rope(cfg_.rope_theta);
        // PHASE MATRIX W_phi = alpha * P (BUG FIX 2026-10-21).
        //
        // The README documents "the phase activation is a modulation on top,
        // applied with W_phi = α·P where P is the model's X-factor projector".
        // No code path ever called set_phase_matrix, so `has_phi` stayed false,
        // the kernel ran with have_phi=0, and set_phase_alpha() had EXACTLY ZERO
        // effect — measured: perplexity identical (29.68) for α = 0, 0.1, 0.3,
        // 0.6 and 1.0. The knob was inert.
        //
        // P is the X-factor projector of the model's embedding matrix: the
        // orthogonal projector onto the manifold where the token embeddings
        // live (winnex-xfactor's Φ_X). Using it as the phase generator is the
        // documented construction.
        if (phase_alpha_ != 0.0 && !phase_P_.empty()) {
            std::vector<float> Wphi((size_t)D_Q * D, 0.0f);
            for (int o = 0; o < D_Q; ++o)
                for (int j = 0; j < D; ++j)
                    Wphi[(size_t)o * D + j] = (float)(phase_alpha_ * (double)phase_P_[(size_t)o * D + j]);
            attns[l]->set_phase_matrix(Wphi.data());
        }
        attns[l]->set_attention_bias(W.b_q.empty() ? nullptr : W.b_q.data(),
                                     W.b_k.empty() ? nullptr : W.b_k.data(),
                                     W.b_v.empty() ? nullptr : W.b_v.data());
        if (spectral_alpha_ != 0.0) attns[l]->set_spectral_alpha(spectral_alpha_);
    }

    std::vector<float> logits_all;
    if (all_positions) logits_all.resize((size_t)seq_len * cfg_.vocab_size);

    std::vector<float> h(D);
    for (int t = 0; t < seq_len; ++t) {
        std::memcpy(h.data(), hidden.data() + (size_t)t * D, D * sizeof(float));

        // All layers for this token.
        for (int l = 0; l < cfg_.num_hidden_layers; ++l) {
            const auto& W = layers_[l];

            // 1. Input RMSNorm.
            std::vector<float> hn(D);
            winnex_nano::rms_norm(hn.data(), h.data(), W.in_norm.data(), 1, D, cfg_.rms_norm_eps);

            // 2. Ψ attention (GPU): the HilbertAttention appends this token's
            //    R/H to its cache and returns y = o_proj·(top-k attention).
            std::vector<float> y(D);
            attns[l]->forward_gpu(hn.data(), y.data(), phase_alpha_);

            // 3. Residual.
            for (int i = 0; i < D; ++i) h[i] += y[i];

            // 4. Post norm + MLP (SiLU).
            std::vector<float> hn2(D);
            winnex_nano::rms_norm(hn2.data(), h.data(), W.post_norm.data(), 1, D, cfg_.rms_norm_eps);
            std::vector<float> gate(cfg_.intermediate_size), up(cfg_.intermediate_size), mlp(D);
            winnex_nano::dense_matmul(gate.data(), hn2.data(), W.g_w.data(), D, cfg_.intermediate_size);
            winnex_nano::dense_matmul(up.data(), hn2.data(), W.u_w.data(), D, cfg_.intermediate_size);
            winnex_nano::silu_inplace(gate.data(), cfg_.intermediate_size);
            for (int i = 0; i < cfg_.intermediate_size; ++i) gate[i] *= up[i];
            winnex_nano::dense_matmul(mlp.data(), gate.data(), W.d_w.data(), cfg_.intermediate_size, D);
            for (int i = 0; i < D; ++i) h[i] += mlp[i];
        }

        // Final norm + lm_head → logits for this token.
        std::vector<float> hnorm(D);
        winnex_nano::rms_norm(hnorm.data(), h.data(), final_norm_.data(), 1, D, cfg_.rms_norm_eps);
        std::vector<float> logits(cfg_.vocab_size);
        winnex_nano::dense_matmul(logits.data(), hnorm.data(), embed_tokens_.data(), D, cfg_.vocab_size);
        if (all_positions) {
            std::memcpy(logits_all.data() + (size_t)t * cfg_.vocab_size,
                        logits.data(), cfg_.vocab_size * sizeof(float));
        } else if (t == seq_len - 1) {
            logits_all = std::move(logits);
        }
    }

    for (auto* a : attns) delete a;
    return logits_all;
}

std::vector<float> PsiForwardEngine::forward_next(const std::vector<float>& hidden) {
    // For the benchmark we use forward(hidden, 1) (one token at a time, the
    // HilbertAttention cache persists within the call).
    return forward(hidden, 1, false);
}

} // namespace winnex_xfactor_c
