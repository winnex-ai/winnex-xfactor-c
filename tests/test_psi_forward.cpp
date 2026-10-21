// test_psi_forward.cpp — validates the Ψ forward engine (full model).
//
// Gate F1/F1.5: the 24-layer Ψ forward on the real Qwen2.5-0.5B must:
//   - produce FINITE logits (no NaN/Inf)
//   - produce non-degenerate layer-0 output (std > 0.1)
//   - run a small sequence without crashing
//
// Usage: test_psi_forward <model_dir> [cache_path]
#include "winnex_xfactor_c/psi_forward.hpp"

#include <cmath>
#include <cstdio>
#include <string>

using winnex_xfactor_c::PsiForwardEngine;

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("usage: %s <model_dir> [cache_path]\n", argv[0]);
        return 2;
    }
    std::string model_dir = argv[1];
    std::string cache = argc > 2 ? argv[2] : "";

    int failures = 0;
    try {
        PsiForwardEngine psi(model_dir, cache);
        printf("PsiForward: %d layers, hidden %d, vocab %d\n",
               psi.num_layers(), psi.hidden_size(), psi.vocab_size());

        // Feed a small sequence of zero-ish hidden states (deterministic).
        const int D = psi.hidden_size();
        const int seq = 4;
        std::vector<float> hidden((size_t)seq * D, 0.001f);

        std::vector<float> logits = psi.forward(hidden, seq, false);
        bool finite = true;
        for (float v : logits) if (!std::isfinite(v)) { finite = false; break; }
        printf("forward(seq=%d) → logits[%zu] finite: %s\n", seq, logits.size(),
               finite ? "OK" : "FAIL");
        if (!finite) ++failures;

        double std_ = 0.0, mean = 0.0;
        for (float v : logits) mean += v;
        mean /= (double)logits.size();
        for (float v : logits) std_ += (v - mean) * (v - mean);
        std_ = std::sqrt(std_ / (double)logits.size());
        printf("logits std = %.4f (non-degenerate if > 0.1): %s\n", std_,
               std_ > 0.1 ? "OK" : "FAIL");
        if (std_ <= 0.1) ++failures;
    } catch (const std::exception& e) {
        printf("EXCEPTION: %s\n", e.what());
        ++failures;
    }

    if (failures == 0) {
        printf("\nALL PSI-FORWARD TESTS PASSED\n");
        return 0;
    }
    printf("\n%d FAILURE(S)\n", failures);
    return 1;
}
