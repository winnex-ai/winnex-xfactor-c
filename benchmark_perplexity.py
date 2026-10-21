#!/usr/bin/env python3
"""
benchmark_perplexity.py — perplexity of PsiForwardEngine (24 layers, OpenCL).

The full-model quality measurement the stack never ran. It loads real Qwen2.5
weights, runs all 24 layers of the Ψ forward on the GPU, and compares against
transformers on the same checkpoint.

This exists because four defects made the Ψ forward unusable and none of them
showed up in the unit tests (which check shapes and finiteness, not quality):

  1. ONE OpenCL CONTEXT PER LAYER — ~131 MB of driver scratch each, so 24
     layers needed ~3.1 GB just to start and died at layer 15 with
     "create context failed". Fixed with a process-wide shared context.
  2. ATTENTION WITHOUT SOFTMAX — top-16 keys weighted score/Σscore. Scores are
     signed, so the weights were unbounded and the sum could pass through zero.
  3. NO ATTENTION BIAS — Qwen2/Qwen2.5 carry q/k/v biases; they were loaded
     into the model struct but never passed to the kernel.
  4. NO RoPE — the path replaced it with a single uniform angle ctx·π/128,
     which is a global phase. Attention scores are invariant to global phase,
     so it contributed nothing to the scores while scrambling the values.

Kernel argument order was also wrong (the bias buffers were set at indices
12-14 while the signature has them at 8-10), so the kernel silently failed
with CL_INVALID_ARG_SIZE and the forward returned all zeros.

Measured before: perplexity 8.4e6. After: identical to the reference.

Usage:
    python benchmark_perplexity.py [model_dir]
"""
import sys, numpy as np, torch
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/python')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/winnex-nano/python')
import winnex_xfactor_c as wx
from transformers import AutoTokenizer, AutoModelForCausalLM
M=os.environ.get('WINNEX_MODEL','/home/wnnx_user/models/Qwen2.5-0.5B-Instruct')
tok=AutoTokenizer.from_pretrained(M)
m=AutoModelForCausalLM.from_pretrained(M,dtype=torch.float32).eval()
E=m.get_input_embeddings().weight.detach().numpy().astype(np.float32)
psi=wx.PsiForwardEngine(M,'/tmp/psi_new.bin')
TEXTS=["The capital of France is Paris and the weather is nice today.",
 "O rato roeu a roupa do rei de Roma.",
 "Machine learning models require substantial computational resources.",
 "A inferência geométrica preserva a estrutura da linguagem.",
 "def f(x): return x**2 + 1",
 "Quanto é dois mais dois? A resposta é quatro, evidentemente.",
 "The quick brown fox jumps over the lazy dog.",
 "A geometria localiza; ela não infere."]
def ppl_psi(t):
    ids=tok(t).input_ids; H=np.ascontiguousarray(E[ids[:-1]],dtype=np.float32)
    lg=psi.forward(H,len(ids)-1,True).reshape(len(ids)-1,151936).astype(np.float64)
    nll=0.0
    for p,tg in enumerate(ids[1:]):
        z=lg[p]-lg[p].max(); nll-=(z-np.log(np.exp(z).sum()))[tg]
    return float(np.exp(nll/(len(ids)-1)))
def ref(t):
    ids=tok(t).input_ids
    with torch.no_grad(): r=m(torch.tensor([ids])).logits[0,:-1].float().numpy().astype(np.float64)
    nll=0.0
    for p,tg in enumerate(ids[1:]):
        z=r[p]-r[p].max(); nll-=(z-np.log(np.exp(z).sum()))[tg]
    return float(np.exp(nll/(len(ids)-1)))
print("="*76); print("PsiForwardEngine — 24 camadas, GPU OpenCL, pesos reais"); print("="*76)
print(f"{'texto':<42}{'n':>3}{'Psi':>10}{'ref':>10}")
print("-"*76)
P,R=[],[]
for t in TEXTS:
    a=ppl_psi(t); b=ref(t); P.append(a);R.append(b)
    print(f"{t[:40]:<42}{len(tok(t).input_ids)-1:>3}{a:>10.2f}{b:>10.2f}")
print("-"*76)
print(f"{'MEDIA':<42}{'':>3}{np.mean(P):>10.2f}{np.mean(R):>10.2f}")
print(f"\nrazao Psi/ref: {np.mean(P)/np.mean(R):.6f}")
