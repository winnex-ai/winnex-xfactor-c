# winnex-xfactor-c

**Native Ψ attention engine — latent coupling + phase activation, adaptive to pretrained LLMs (no training).**

The classical QKV attention block is replaced by the Ψ mechanism:

```
Z   = W_z·x          (latent coupling, d_latent = 4·d)
Q/R/H = W_q'/r'/h'·Z (derived heads from the shared latent)
Ψ(v) = v ⊙ exp(i·W_φ·v)              (phase activation)
F(k) = exp(i·α·arctan(ln|k|))        (spectral log-phase filter, optional)
pos  = ctx·π/128     (additive positional phase — replaces RoPE, PiPrime step)
scores = Re(Q'·R'*)   + Madhava top-k (ksel=16, weight = score/Σscore, NO softmax)
out = scores·H → o_proj
```

**Position WITHOUT RoPE:** each token at position `ctx` gets an additive phase
`ctx·π/128` on its complex Q/R (the spectral-tokenizer principle: position is
encoded in the phase). The step uses a prime-scale θ (PiPrime) so distinct
positions have distinguishable phases — no transformer positional machinery.
Validated: the same input at positions 0/1/2 produces distinguishable outputs,
monotonic in distance.

The attention uses the **Madhava principle** (exactly like winnex-nano forward.cpp):
selective top-k by score, weight = score/sum(top-k), no softmax, no exp.

C++20 core, pybind11 bindings, reuses the [winnex-nano](https://pypi.org/project/winnex-nano/)
native sources (safetensors, tensor_ops, x_factor, spectral_tokenizer) without duplication.

## Adaptability (the key property)

The weights `W_z, W_q', W_r', W_h'` are **derived deterministically** from a
pretrained model's `Q/K/V` projections via a pseudoinverse:

```
W_z = [ I ; R₁ ; R₂ ; R₃ ]  ∈ ℝ^{4d×d}   (orthonormal blocks, RᵢᵀRᵢ = I)
W_zᵀ·W_z = I + R₁ᵀR₁ + R₂ᵀR₂ + R₃ᵀR₃ = 4·I_d      (W_z is TALL 4d×d)
W_q' = (1/4)·W_q·W_zᵀ,  W_r' = (1/4)·W_k·W_zᵀ,  W_h' = (1/4)·W_v·W_zᵀ
```

**Preservation (exact):** `W_q'·W_z = (1/4)·W_q·(W_zᵀ·W_z) = W_q`. With `W_φ = 0`,
the Ψ attention computes **identical Q/R/H to the classical QKV** — measured
error ~2e-8 (relative Frobenius). The phase activation is a **modulation on
top**, applied with `W_φ = α·P` where `P` is the model's X-factor projector
(deterministic).

## The Ψ attention (OpenCL Hilbert space)

`HilbertAttention` implements the full attention on a generic GPU via
runtime-loaded OpenCL (dlopen + JIT kernels, the winnex-madhava pattern —
**no CUDA**, the Quake/GLSL model):

```
scores[q][k] = Re(⟨Q'_q, R'_k⟩) = Σ_j (Q_re·R_re + Q_im·R_im)   (Hilbert product)
scores = L2_normalize(scores, per head)                          (no softmax)
out[h][j] = Σ_k scores[h][k] · H[k][kvh][j]                      (aggregate values)
y = o_proj · out
```

The KV cache lives **on the device** (R/H buffers grown by doubling). The phase
is applied ONCE at latent time; the cached complex key IS the phase-activated
state — attention never recomputes cos/sin. OpenCL is REQUIRED (no CPU fallback).

**Validated on the RTX 5060 Ti** (OpenCL, no CUDA, no LD_PRELOAD needed):
- Kernel correctness vs CPU reference: `max|out − ref| = 2e-7` (float32)
- Preservação no modelo real: `‖W'·W_z − W‖ = 2.36e-8`

## GPU kernels (fused, 4 per token)

The Ψ pipeline is **4 kernels per token**, Q/R/H stay on the device:

1. `latent_z` — Z = W_z·x (grid 4d)
2. `latent_qrh_phase` — Q/R/H from Z + phase, **appended DIRECTLY to the device cache**
3. `attn_fused` — Madhava top-k (scores → top-16 → weight=score/Σ) in ONE kernel
4. `attn_outproj` — o_proj + single readback

Spectral log-phase filter `F(k)=exp(i·α·arctan(ln|k|))` (opt-in, `set_spectral_alpha`)
and the classical QKV GPU path (`set_classical_weights`/`forward_gpu_classical`,
same Madhava top-k — a fair comparison) are also implemented.

## Status

- **F0** — package skeleton + native build: ✅
- **F1** — `derive_weights` (QKV → latent coupling, exact): ✅ (`‖W'·W_z − W‖ < 1e-3`, measured ~2e-8)
- **F2** — `latent_coupling` + phase score: ✅ (phase off == classical dot; phase on == `Σ Q·R·cos(φq−φr)`)
- **F3** — `HilbertAttention` (OpenCL, 4 fused kernels, Madhava top-k): ✅ (RTX 5060 Ti)
- **F4** — spectral log-phase filter (opt-in): ✅ (`test_spectral` — isolated, bug fixed)
- **F5** — additive positional phase (replaces RoPE, PiPrime step): ✅ (`test_position`)
- **F6** — honest AC benchmark on Qwen2.5-0.5B: ✅ (`benchmark_ac.py`)

## Build

```bash
cmake -S . -B build && cmake --build build -j
# Python binding (from repo root):
c++ -O3 -std=c++20 -fPIC -shared \
  -I include -I ../winnex-nano/include \
  -I $(python3 -c "import pybind11; print(pybind11.get_include())") \
  $(python3-config --includes) \
  bindings/py_module.cpp src/derive_weights.cpp src/latent_coupling.cpp \
  ../winnex-nano/src/tensor_ops.cpp ../winnex-nano/src/x_factor.cpp \
  ../winnex-nano/src/spectral_tokenizer.cpp ../winnex-nano/src/safetensors.cpp \
  -o python/winnex_xfactor_c/_winnex_xfactor_c$(python3-config --extension-suffix)
```

## Quick start

```python
import numpy as np
import winnex_xfactor_c as wx

W_q = np.random.randn(1024, 512).astype(np.float32)  # (d_q, d)
W_k = np.random.randn(256, 512).astype(np.float32)   # (d_kv, d)
W_v = np.random.randn(256, 512).astype(np.float32)

W_z, W_qp, W_rp, W_hp = wx.derive_from_qkv(W_q, W_k, W_v)
err = wx.preservation_error(W_q, W_z, W_qp)   # ~2e-8  (exact reparametrization)

x = np.random.randn(512).astype(np.float32)
Z, Q, R, H = wx.latent_forward(x, W_z, W_qp, W_rp, W_hp, 32, 8, 32)  # phase off
```

## Honest limitations

- The ΨQRH framework this is based on is **CONCEPTUAL** (per project rule #1).
- The original benchmark numbers (perplexity 19.8 → 6.6, "-66.7%") compare models
  of different sizes (3.3M vs 21.7M params) — NOT attributable to the architecture.
- The FFT + log-phase filter is a **regularizer / circular convolution**, not
  "quantum physics". Its cost and benefit must be measured (F4/F5), not assumed.
- Phase activation and the spectral path may be slower than the classical block
  (complex arithmetic + 4× latent). Speed is measured in F5.

## License

Business Source License 1.1 (BSL 1.1) — `pay@winnex.ai`.

(c) Winnex Brasil Soluções Empresariais LTDA-ME
