"""
winnex-xfactor-c — native Ψ attention engine (latent coupling + phase).

Deterministic, training-free adaptation of a pretrained LLM to the Ψ attention
mechanism. Replaces the classical QKV block with:

    Z   = W_z·x          (latent coupling, d_latent = 4·d)
    Q/R/H = W_q'/r'/h'·Z (derived heads from the shared latent)
    Ψ(v) = v ⊙ exp(i·W_φ·v)              (phase activation)
    scores = Re(Q'·R'*)   + L2 norm      (no softmax)
    out = scores·H → o_proj

Note: latent_forward returns Q/R as COMPLEX interleaved pairs (2× length).

The weights W_z, W_q', W_r', W_h' are DERIVED from the pretrained QKV via a
pseudoinverse (exact reparametrization when W_φ = 0). No training. C++20 core,
pybind11 bindings; numpy only for the array interface.

BSL 1.1 | pay@winnex.ai | (c) Winnex Brasil Soluções Empresariais LTDA-ME
"""

from __future__ import annotations

import os
import sys

_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

try:
    from . import _winnex_xfactor_c as _native
except ImportError:
    try:
        import _winnex_xfactor_c as _native  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "winnex_xfactor_c native extension not found. Build with: pip install ."
        ) from exc

import numpy as np

# Native objects/functions exposed by the binding.
derive_weights = _native.derive_weights
preservation_error = _native.preservation_error
phase_factors = _native.phase_factors
latent_forward = _native.latent_forward
score_phase = _native.score_phase
HilbertAttention = _native.HilbertAttention
PsiForwardEngine = _native.PsiForwardEngine

__version__ = "0.1.0"
__all__ = [
    "derive_weights",
    "preservation_error",
    "phase_factors",
    "latent_forward",
    "score_phase",
    "HilbertAttention",
    "PsiForwardEngine",
    "__version__",
]


def derive_from_qkv(
    W_q: np.ndarray,
    W_k: np.ndarray,
    W_v: np.ndarray,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derive the latent-coupling weights from the classical Q/K/V projections.

    Parameters
    ----------
    W_q : np.ndarray  (d_q × d) row-major query weight.
    W_k : np.ndarray  (d_kv × d) row-major key weight.
    W_v : np.ndarray  (d_kv × d) row-major value weight.
    seed : int        deterministic rotation seed.

    Returns
    -------
    (W_z, W_q_prime, W_r_prime, W_h_prime)
        W_z       : (4d × d) structural latent projection.
        W_q_prime : (d_q × 4d) derived query.
        W_r_prime : (d_kv × 4d) derived relational (key).
        W_h_prime : (d_kv × 4d) derived value.

    Preservation: ``W_q_prime @ W_z ≈ W_q`` (exact when W_φ = 0).
    """
    arrs = [np.ascontiguousarray(W, dtype=np.float32) for W in (W_q, W_k, W_v)]
    d_q, d = arrs[0].shape
    d_kv = arrs[1].shape[0]
    if arrs[1].shape != (d_kv, d) or arrs[2].shape != (d_kv, d):
        raise ValueError("shapes must be (d_q,d), (d_kv,d), (d_kv,d)")
    return derive_weights(arrs[0], arrs[1], arrs[2], d, d_q, d_kv, seed)


# __version__ vem dos metadados do pacote (tag git via scikit-build-core),
# nunca de um literal — mesmo padrão de winnex-nano e winnex-xfactor. Um valor
# hardcoded já divergiu do publicado nas outras peças da stack, e o erro é
# silencioso: o build passa e o wheel sobe com a versão errada.
try:
    from importlib.metadata import version as _v, PackageNotFoundError as _PNF
    try:
        __version__ = _v("winnex-xfactor-c")
    except _PNF:
        __version__ = "0.0.0+local"
except Exception:  # pragma: no cover
    __version__ = "0.0.0+local"
