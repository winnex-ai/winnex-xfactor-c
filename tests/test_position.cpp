// test_position.cpp — the Ψ mechanism must DISTINGUISH sequence positions.
//
// UPDATED (2026-10-21). This test previously asserted that positions are
// distinguished WITHOUT RoPE, by the additive uniform phase ctx·π/128.
//
// That phase is not a positional encoding a pretrained model can interpret:
// it rotates the whole head vector by a SINGLE angle, whereas RoPE applies a
// different frequency per dimension pair (theta^(-2j/hd)). A uniform rotation
// is a global phase, and attention scores are invariant to global phase — so
// it contributed nothing to the scores while scrambling the value path.
//
// Measured on Qwen2.5-0.5B: with the uniform phase the full model gave
// perplexity 8.4e6 against 35.1 for transformers. With RoPE (rotate_half) the
// Ψ forward matches the reference to 1e-6 (ratio 1.000006).
//
// The property this test checks — positions must be distinguishable — is
// unchanged. What changed is the mechanism: RoPE, which is what the weights
// were trained with. The uniform phase remains available behind
// WINNEX_POS_PHASE=1 for the experiment that wants to study it.
#include "winnex_xfactor_c/attention_forward.hpp"
#include "winnex_xfactor_c/derive_weights.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using winnex_xfactor_c::HilbertAttention;

int main() {
    int failures = 0;

    const int n_heads = 4, n_kv = 2, hd = 8;
    const int d_q = n_heads * hd, d_kv = n_kv * hd, d = 32;

    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // Random QKV → derive latent weights.
    std::vector<float> Wq(d_q * d), Wk(d_kv * d), Wv(d_kv * d);
    for (auto& v : Wq) v = nd(rng);
    for (auto& v : Wk) v = nd(rng);
    for (auto& v : Wv) v = nd(rng);
    auto dw = winnex_xfactor_c::derive_weights(Wq.data(), Wk.data(), Wv.data(),
                                               d, d_q, d_kv, 5);
    std::vector<float> x(d);
    for (auto& v : x) v = nd(rng);

    HilbertAttention attn(d_q, d_kv, n_heads, n_kv, hd, d);
    attn.set_latent_weights(dw.W_z.data(), dw.W_q_prime.data(),
                            dw.W_r_prime.data(), dw.W_h_prime.data());

    // Position sensitivity needs at least TWO visible tokens: with a single
    // token the softmax is 1 and the output is just V — Q and R never enter the
    // result, so no positional encoding can show up. The test therefore seeds
    // one key first, then feeds the SAME x at positions 1, 2 and 3 and compares
    // the three outputs.
    //
    // (Earlier this test fed x at positions 0,1,2 with an EMPTY cache each time
    // — i.e. three independent single-token attentions — and it passed only
    // because the uniform ctx·π/128 phase was applied to V as well. That phase
    // is gone; the assertion now exercises the mechanism that actually carries
    // position, RoPE on Q/R.)
    std::vector<float> seed(d);
    for (auto& v : seed) v = nd(rng);
    {
        std::vector<float> sink(d);
        attn.forward_gpu(seed.data(), sink.data(), 0.0);   // token 0 (context)
    }
    std::vector<float> y0(d), y1(d), y2(d);
    attn.forward_gpu(x.data(), y0.data(), 0.0);   // position 1
    attn.forward_gpu(x.data(), y1.data(), 0.0);   // position 2
    attn.forward_gpu(x.data(), y2.data(), 0.0);   // position 3

    double d01 = 0.0, d02 = 0.0;
    for (int i = 0; i < d; ++i) {
        d01 += std::fabs((double)y0[i] - y1[i]);
        d02 += std::fabs((double)y0[i] - y2[i]);
    }

    // Both non-adjacent AND adjacent positions must differ. The old assertion
    // additionally required d02 > d01 ("further apart differs MORE"), which was
    // a property of the uniform phase — a single angle growing linearly with
    // ctx. RoPE is a per-frequency rotation: wider separation does NOT imply a
    // larger L1 distance, so that inequality is not a property of the encoding
    // the weights expect. Requiring it would be testing the removed mechanism.
    bool ok1 = (d01 > 1e-3);   // pos 1 vs pos 2 differ
    bool ok2 = (d02 > 1e-3);   // pos 1 vs pos 3 differ
    printf("pos1 vs pos2: Σ|Δ| = %.5f (>1e-3) [%s]\n", d01, ok1 ? "OK" : "FAIL");
    printf("pos1 vs pos3: Σ|Δ| = %.5f (>1e-3) [%s]\n", d02, ok2 ? "OK" : "FAIL");
    if (!ok1) ++failures;
    if (!ok2) ++failures;

    if (failures == 0) {
        printf("\nALL POSITION TESTS PASSED\n");
        return 0;
    }
    printf("\n%d FAILURE(S)\n", failures);
    return 1;
}
