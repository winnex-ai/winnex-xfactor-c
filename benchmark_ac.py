#!/usr/bin/env python3
"""
benchmark_ac.py — honest benchmark of the AC (adaptive spectral) component.

Model: Qwen2.5-0.5B-Instruct (real, dense, compatible with winnex-nano).

What is measured HONESTLY (no fabricated perplexity of an unintegrated forward):
  [1] BASELINE BEHAVIOR  — the full 24-layer model via ForwardEngine (winnex-nano)
      on real token embeddings. This confirms the model runs and gives a
      reference for the hidden-state scale.
  [2] AC EFFECT on layer 0 — the HilbertAttention (Ψ latent coupling, Madhava
      top-k) is applied to layer 0 of the model with the REAL weights. We
      measure how the spectral filter (F(k)=exp(i·α·arctan(ln|k|))) changes the
      layer-0 attention output as a function of α:
        - output norm and scale
        - delta vs α=0 (the Ψ latent coupling WITHOUT spectral)
        - relative delta (how much the AC perturbs the layer output)

Modes compared (all on the SAME layer-0 real input):
  - α=0   : Ψ latent coupling, no spectral (mathematically == classical QKV)
  - α=0.3 : + spectral log-phase filter
  - α=0.6 : + stronger spectral

Honesty: this is a SINGLE-layer effect (layer 0) on real weights, reported as
such. It shows whether and how much the AC changes the model's layer output —
NOT a claim of improved perplexity (that requires Ψ integrated into all 24
layers, a separate step).
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "python"))
sys.path.insert(0, os.path.join(HERE, "../winnex-nano/python"))
sys.path.insert(0, os.path.join(HERE, "../winnex-nano"))

import winnex_xfactor_c as wx
import winnex_nano as wn

MODEL_DIR = os.environ.get(
    "QWEN_MODEL_DIR", "/home/wnnx_user/models/Qwen2.5-0.5B-Instruct"
)
D, D_Q, D_KV = 896, 896, 128
NH, NKV, HD = 14, 2, 64


def rmsnorm(x, w, eps=1e-6):
    return x * w * np.sqrt(1.0 / (np.mean(x * x) + eps))


def layer0_ac_effect():
    """AC effect on layer 0 via HilbertAttention (real weights)."""
    st = wn.Safetensors(os.path.join(MODEL_DIR, "model.safetensors"))
    L = "model.layers.0.self_attn."
    W_q = np.asarray(st.to_float(L + "q_proj.weight"), np.float32).reshape(D_Q, D)
    W_k = np.asarray(st.to_float(L + "k_proj.weight"), np.float32).reshape(D_KV, D)
    W_v = np.asarray(st.to_float(L + "v_proj.weight"), np.float32).reshape(D_KV, D)
    o_proj = np.asarray(st.to_float(L + "o_proj.weight"), np.float32).reshape(D, D_Q)
    in_norm = np.asarray(st.to_float("model.layers.0.input_layernorm.weight"), np.float32)

    # Adapt the model's real QKV → latent coupling (the core claim).
    W_z, W_qp, W_rp, W_hp = wx.derive_from_qkv(W_q, W_k, W_v)
    W_zc = np.ascontiguousarray(W_z); W_qpc = np.ascontiguousarray(W_qp)
    W_rpc = np.ascontiguousarray(W_rp); W_hpc = np.ascontiguousarray(W_hp)
    err = wx.preservation_error(W_q, W_z, W_qp, 128)
    print(f"  preservação ‖W'·W_z − W‖ = {err:.2e}")

    # Real token embeddings as the layer input (with input_layernorm).
    emb = wn.sample_embeddings(st, "model.embed_tokens.weight", 8, 42)
    xs = []
    for e in emb:
        xs.append(rmsnorm(e.astype(np.float32), in_norm))

    results = {}
    y0 = None
    for alpha in (0.0, 0.3, 0.6):
        psi = wx.HilbertAttention(D_Q, D_KV, NH, NKV, HD, D)
        psi.set_latent_weights(W_zc, W_qpc, W_rpc, W_hpc)
        psi.set_output_projection(o_proj)
        if alpha != 0.0:
            psi.set_spectral_alpha(alpha)
        # Prefill the cache with the context tokens, then read one output.
        for e in xs:
            psi.forward_gpu(e)
        y = psi.forward_gpu(xs[0])
        norm = float(np.linalg.norm(y))
        entry = {"out_norm": norm, "out_sum": float(y.sum())}
        if alpha == 0.0:
            y0 = y
        else:
            d = float(np.linalg.norm(y - y0))
            entry["delta_vs_alpha0"] = d
            entry["rel_delta"] = float(d / (np.linalg.norm(y0) + 1e-12))
        results[f"alpha_{alpha}"] = entry
    return results


def baseline_check():
    """Confirm the full model runs (ForwardEngine) — the reference for scale."""
    cfg = wn.load_config(os.path.join(MODEL_DIR, "config.json"))
    st = wn.Safetensors(os.path.join(MODEL_DIR, "model.safetensors"))
    fe = wn.ForwardEngine(cfg, st)
    x = wn.sample_embeddings(st, "model.embed_tokens.weight", 1, 100)[0]
    logits = fe.forward(x, 1)
    return {"logits_finite": bool(np.isfinite(logits).all()),
            "logits_std": float(logits.std()),
            "argmax": int(np.argmax(logits))}


def main():
    print("=" * 72)
    print("  WINNEX-XFACTOR-C — BENCHMARK AC (adaptive spectral) honesto")
    print(f"  Modelo: Qwen2.5-0.5B-Instruct | layer 0 | OpenCL GPU")
    print("=" * 72)

    print("\n[1] Baseline (ForwardEngine full 24-layer model):")
    b = baseline_check()
    print(f"    logits finite={b['logits_finite']}  std={b['logits_std']:.3f}  "
          f"argmax={b['argmax']}")

    print("\n[2] AC effect on layer 0 (HilbertAttention, real weights):")
    t0 = time.perf_counter()
    r = layer0_ac_effect()
    dt = time.perf_counter() - t0
    for k, v in r.items():
        d = v.get("delta_vs_alpha0", "-")
        rd = v.get("rel_delta", "-")
        print(f"    {k:<9} out_norm={v['out_norm']:.4f}  "
              f"Δvsα0={d if isinstance(d, str) else f'{d:.6f}'}  "
              f"rel={rd if isinstance(rd, str) else f'{rd:.2%}'}")
    print(f"    (layer-0 forward: {dt:.2f}s)")

    out = os.path.join(HERE, "benchmark_ac.results.json")
    with open(out, "w") as f:
        json.dump({"model": "Qwen2.5-0.5B-Instruct", "layer": 0,
                   "baseline": b, "modes": r}, f, indent=2)
    print(f"\nsaved: {out}")
    print("\nHONESTY: layer-0 single-layer AC effect on real weights. NOT the")
    print("full-model ppl (that needs Ψ in all 24 layers — a separate step).")


if __name__ == "__main__":
    main()
