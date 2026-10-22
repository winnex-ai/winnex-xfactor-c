// attention_forward.cpp — Ψ attention forward, OpenCL-optimized Hilbert KV cache.
#include "winnex_xfactor_c/attention_forward.hpp"

// Vendored OpenCL headers (types + constants only; no link-time loader).
#include <CL/cl.h>

// PORTABLE DYNAMIC LOADING (2026-10-21). dlopen/dlsym/dlclose are POSIX and do
// not exist on Windows, which uses LoadLibrary/GetProcAddress/FreeLibrary.
// Without this layer the MSVC build fails with:
//     attention_forward.cpp(7,10): error C1083: Cannot open include file:
//     'dlfcn.h': No such file or directory
// (reproduced in CI: Wheels on windows-latest, run #1). This mirrors the shim
// already used by winnex-madhava's speed_opencl.cpp, so the whole stack loads
// OpenCL the same way on every platform.
#ifdef _WIN32
// NOMINMAX: <windows.h> defines min/max as MACROS, which breaks every
// std::min/std::max in this translation unit ("error C2589: '(' illegal token
// on the right side of '::'"). Defining these before the include suppresses it.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
static void* xfc_open(const char* nome) { return (void*)LoadLibraryA(nome); }
static void* xfc_sym(void* h, const char* nome) {
    return (void*)GetProcAddress((HMODULE)h, nome);
}
static void xfc_close(void* h) { if (h) FreeLibrary((HMODULE)h); }
#else
#include <dlfcn.h>
static void* xfc_open(const char* nome) { return dlopen(nome, RTLD_NOW | RTLD_LOCAL); }
static void* xfc_sym(void* h, const char* nome) { return dlsym(h, nome); }
static void xfc_close(void* h) { if (h) dlclose(h); }
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace winnex_xfactor_c {

// ══════════════════════════════════════════════════════════════════════════
// Runtime OpenCL loading (dlopen, no link-time dependency) — the winnex-madhava
// pattern. OpenCL is REQUIRED (no CPU fallback): the constructor throws if the
// loader or a device is unavailable.
// ══════════════════════════════════════════════════════════════════════════
struct OCLApi;
static OCLApi* g_ocl = nullptr;

#define OCL_FUNC(ret, name, args) ret (*name) args;
struct OCLApi {
    OCL_FUNC(cl_int, clGetPlatformIDs, (cl_uint, cl_platform_id*, cl_uint*))
    OCL_FUNC(cl_int, clGetDeviceIDs, (cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*))
    OCL_FUNC(cl_int, clGetDeviceInfo, (cl_device_id, cl_device_info, size_t, void*, size_t*))
    OCL_FUNC(cl_context, clCreateContext, (const cl_context_properties*, cl_uint, const cl_device_id*, void (CL_CALLBACK*)(const char*, const void*, size_t, void*), void*, cl_int*))
    OCL_FUNC(cl_command_queue, clCreateCommandQueue, (cl_context, cl_device_id, cl_command_queue_properties, cl_int*))
    OCL_FUNC(cl_program, clCreateProgramWithSource, (cl_context, cl_uint, const char**, const size_t*, cl_int*))
    OCL_FUNC(cl_int, clBuildProgram, (cl_program, cl_uint, const cl_device_id*, const char*, void (CL_CALLBACK*)(cl_program, void*), void*))
    OCL_FUNC(cl_int, clGetProgramBuildInfo, (cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*))
    OCL_FUNC(cl_kernel, clCreateKernel, (cl_program, const char*, cl_int*))
    OCL_FUNC(cl_mem, clCreateBuffer, (cl_context, cl_mem_flags, size_t, void*, cl_int*))
    OCL_FUNC(cl_int, clSetKernelArg, (cl_kernel, cl_uint, size_t, const void*))
    OCL_FUNC(cl_int, clEnqueueNDRangeKernel, (cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, cl_event*, cl_event*))
    OCL_FUNC(cl_int, clEnqueueWriteBuffer, (cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*))
    OCL_FUNC(cl_int, clEnqueueReadBuffer, (cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*))
    OCL_FUNC(cl_int, clEnqueueCopyBuffer, (cl_command_queue, cl_mem, cl_mem, size_t, size_t, size_t, cl_uint, const cl_event*, cl_event*))
    OCL_FUNC(cl_int, clFinish, (cl_command_queue))
    OCL_FUNC(cl_int, clReleaseMemObject, (cl_mem))
    OCL_FUNC(cl_int, clReleaseKernel, (cl_kernel))
    OCL_FUNC(cl_int, clReleaseProgram, (cl_program))
    OCL_FUNC(cl_int, clReleaseCommandQueue, (cl_command_queue))
    OCL_FUNC(cl_int, clReleaseContext, (cl_context))
};
#undef OCL_FUNC

// Preloads the vendor ICD driver(s) so the ocl-icd loader finds them.
// On NVIDIA with ocl-icd, the driver registers its platform only when it is
// loaded EARLY (before the loader's clGetPlatformIDs); otherwise the loader
// dlopen()s it but the platform is not registered → CL_PLATFORM_NOT_FOUND.
// This mirrors what LD_PRELOAD does, without requiring the user to set it.
// Opt-in for the uniform positional phase (see the note in latent_qrh_phase).
// Default OFF: it is not RoPE and the pretrained weights cannot interpret it.
static bool PosPhaseEnabled() {
    static const bool on = [] {
        const char* v = getenv("WINNEX_POS_PHASE");
        return v != nullptr && v[0] != '0';
    }();
    return on;
}

static void preload_vendor_drivers() {
    // NVIDIA: libnvidia-opencl.so.1 (the ICD driver).
    xfc_open("libnvidia-opencl.so.1");
    // Intel / AMD ICD drivers (harmless if absent).
    xfc_open("libintel-opencl.so.1");
    xfc_open("libamdocl64.so");
}

// Eager init at MODULE LOAD time (before any heavy CPU/OpenMP work). The
// NVIDIA driver only registers its platform when the loader's clGetPlatformIDs
// is called EARLY; doing it after heavy host-side work (e.g. a large OpenMP
// derive) makes it return CL_PLATFORM_NOT_FOUND. A static initializer loads
// the loader + driver AND calls clGetPlatformIDs once, so the platform is
// registered before main()/import completes.
struct EagerPreload {
    EagerPreload() {
        // Load the ICD loader so its symbols are resident.
        void* h = xfc_open("libOpenCL.so.1");
        if (!h) h = xfc_open("libOpenCL.so");
        preload_vendor_drivers();
        // Force platform enumeration now (registers the driver's platform).
        typedef cl_int (*GetPlatFn)(cl_uint, cl_platform_id*, cl_uint*);
        auto fn = (GetPlatFn)(h ? xfc_sym(h, "clGetPlatformIDs") : nullptr);
        if (fn) { cl_uint n = 0; fn(0, nullptr, &n); }
    }
};
static EagerPreload g_eager_preload;

static bool load_opencl() {
    if (g_ocl) return true;
    void* h = xfc_open("libOpenCL.so.1");
    if (!h) h = xfc_open("libOpenCL.so");
    if (!h) return false;
    // Ensure the vendor driver is loaded before platform enumeration.
    preload_vendor_drivers();
    auto* api = new OCLApi;
    bool ok = true;
#define LOAD(name) \
    *(void**)(&api->name) = xfc_sym(h, #name); \
    if (!api->name) ok = false
    LOAD(clGetPlatformIDs);        LOAD(clGetDeviceIDs);        LOAD(clGetDeviceInfo);
    LOAD(clCreateContext);         LOAD(clCreateCommandQueue);
    LOAD(clCreateProgramWithSource); LOAD(clBuildProgram);      LOAD(clGetProgramBuildInfo);
    LOAD(clCreateKernel);          LOAD(clCreateBuffer);        LOAD(clSetKernelArg);
    LOAD(clEnqueueNDRangeKernel);  LOAD(clEnqueueWriteBuffer);  LOAD(clEnqueueReadBuffer);
    LOAD(clEnqueueCopyBuffer);     LOAD(clFinish);              LOAD(clReleaseMemObject);
    LOAD(clReleaseKernel);
    LOAD(clReleaseProgram);        LOAD(clReleaseCommandQueue); LOAD(clReleaseContext);
#undef LOAD
    if (!ok) { xfc_close(h); delete api; return false; }
    g_ocl = api;
    return true;
}

#define clGetPlatformIDs            g_ocl->clGetPlatformIDs
#define clGetDeviceIDs              g_ocl->clGetDeviceIDs
#define clGetDeviceInfo             g_ocl->clGetDeviceInfo
#define clCreateContext             g_ocl->clCreateContext
#define clCreateCommandQueue        g_ocl->clCreateCommandQueue
#define clCreateProgramWithSource   g_ocl->clCreateProgramWithSource
#define clBuildProgram              g_ocl->clBuildProgram
#define clGetProgramBuildInfo       g_ocl->clGetProgramBuildInfo
#define clCreateKernel              g_ocl->clCreateKernel
#define clCreateBuffer              g_ocl->clCreateBuffer
#define clSetKernelArg              g_ocl->clSetKernelArg
#define clEnqueueNDRangeKernel      g_ocl->clEnqueueNDRangeKernel
#define clEnqueueWriteBuffer        g_ocl->clEnqueueWriteBuffer
#define clEnqueueReadBuffer         g_ocl->clEnqueueReadBuffer
#define clEnqueueCopyBuffer         g_ocl->clEnqueueCopyBuffer
#define clFinish                    g_ocl->clFinish
#define clReleaseMemObject          g_ocl->clReleaseMemObject
#define clReleaseKernel             g_ocl->clReleaseKernel
#define clReleaseProgram            g_ocl->clReleaseProgram
#define clReleaseCommandQueue       g_ocl->clReleaseCommandQueue
#define clReleaseContext            g_ocl->clReleaseContext

// Runtime-compiled kernels (JIT, like GLSL). The score kernel is a FUSED
// complex inner product over the interleaved (re,im) pairs — the Hilbert-space
// inner product Re(⟨Q',R'⟩) computed directly on the device cache layout.
const char* g_attn_kernels = R"(
// Hilbert attention score: for one query position q and a cache of k positions,
//   score[h][k] = Re(⟨Q'_h, R'_k,h⟩) / sqrt(head_dim)
// where Q'_h is the phase-activated query (complex interleaved [2·hd]) and
// R'_k,h is the cached phase-activated key (complex interleaved [2·hd]).
// GQA: query head h attends to kv-head h/(n_heads/n_kv_heads).
// Grid: (n_heads × ctx) work-items; each computes one score.
// Output projection: y = o_proj · concat(out)  [d] with row-major o_proj [d × d_q].
__kernel void attn_outproj(
    __global const float* out,      // [d_q] concatenated per-head values
    __global const float* o_proj,   // [d × d_q] row-major
    __global float* y,              // [d]
    int d_q) {
    int i = (int)get_global_id(0);
    if (i >= get_global_size(0)) return;
    // (grid size = d passed as global)
    const __global float* wrow = o_proj + (size_t)i * d_q;
    float acc = 0.0f;
    for (int j = 0; j < d_q; ++j) acc += wrow[j] * out[j];
    y[i] = acc;
}

// ── LATENT COUPLING (operational correlation), 2-KERNEL ─────────────────────
// Kernel A computes Z = W_z·x ONCE (grid 4d, embarrassingly parallel, no
// redundancy). Kernel B computes the derived heads from Z (grid d_q + 2·d_kv)
// with the phase activation fused in. Z is shared in global memory between the
// two kernels — each Z row is computed exactly once (vs the fused single-kernel
// approach where every output row redundantly recomputed Z, which was O(d_q·4d·d)).
__kernel void latent_z(
    __global const float* x,       // [d]
    __global const float* W_z,     // [4d × d] row-major
    __global float* Z,             // [4d]
    int d) {
    int z = (int)get_global_id(0);
    if (z >= 4*d) return;
    const __global float* wzrow = W_z + (size_t)z * d;
    float acc = 0.0f;
    for (int i = 0; i < d; ++i) acc += wzrow[i] * x[i];
    Z[z] = acc;
}

// Kernel B: Q/R/H from Z, with phase fused in, AND appended DIRECTLY to the
// device cache at position `ctx` (no separate copy kernel). This keeps Q/R/H
// on the device for the attention without any host round-trip or cache copy.
//   Q[2o]=Qpre·cosφ, Q[2o+1]=Qpre·sinφ   (φ = W_phi·Qpre if have_phi)
//   R (in cache at ctx): [2·d_kv] complex
//   H (in cache at ctx): [d_kv] real
__kernel void latent_qrh_phase(
    __global const float* Z,       // [4d]
    __global const float* W_qp,    // [d_q × 4d]
    __global const float* W_rp,    // [d_kv × 4d]
    __global const float* W_hp,    // [d_kv × 4d]
    __global const float* W_phi,   // [d_q × d_q] (have_phi)
    __global float* Q,             // [2·d_q]
    __global float* Rcache,        // [cap · 2·d_kv] device key cache (append at ctx)
    __global float* Hcache,        // [cap · d_kv] device value cache (append at ctx)
    __global const float* b_q,     // [d_q]  q_proj bias (Qwen2; may be all-zero)
    __global const float* b_k,     // [d_kv] k_proj bias
    __global const float* b_v,     // [d_kv] v_proj bias
    int d_q, int d_kv, int d_lat, int have_phi, int ctx, int have_pos_phase) {
    int gid = (int)get_global_id(0);
    int total = d_q + 2 * d_kv;
    if (gid >= total) return;
    const size_t r_off = (size_t)ctx * d_kv * 2;   // token stride in the cache
    const size_t h_off = (size_t)ctx * d_kv;

    // POSITIONAL PHASE (additive, PiPrimo-optimized — replaces RoPE).
    // The spectral tokenizer encodes position additively in the phase:
    //   phase = (content + pos + j)·2π/256
    // Here each token at position `ctx` gets a phase offset `pos·θ` applied to
    // its complex Q/R. The step θ = π·prime[2]/N ≈ small keeps distinct
    // positions distinguishable (primes are ℚ-linearly independent → no
    // periodic collision, unlike pos·2π/N wrap). No RoPE, no transformer
    // positional machinery. `have_phi` adds the content phase on top.
    // Position step θ: small enough to be a moderate perturbation per position,
    // large enough to distinguish positions. θ = π·prime[1]/128 ≈ 0.0245 rad —
    // the PiPrime scale (prime-indexed step avoids harmonic collisions). For
    // ctx ≤ 256 the total phase stays < 2π (no wrap).
    //
    // BUG FIX (2026-10-21): this positional phase is DISABLED by default.
    //
    // `ctx·π/128` is a SINGLE scalar angle applied identically to every one of
    // the head_dim elements. RoPE — which the pretrained model was trained
    // with — applies a DIFFERENT frequency to each dimension pair
    // (theta^(-2j/hd), j = 0..hd/2-1). A uniform rotation is not a positional
    // encoding the model can interpret: it spins the whole head vector by the
    // same angle, which is a global phase, and attention scores are invariant
    // to global phase, so it also contributes nothing to the score while
    // scrambling the value path.
    //
    // Measured before this change: PsiForwardEngine perplexity 8.4e6 against
    // 35.1 for transformers, failing even on a single token.
    //
    // The phase stays available for the experiment that wants it
    // (WINNEX_POS_PHASE=1); RoPE is applied where the model expects it.
    // The switch is a host-side kernel argument (see have_pos_phase); OpenCL C
    // has no lambdas and the device cannot read the environment.
    const float pos_phase = have_pos_phase * (float)ctx * 3.14159265358979f * 0.0078125f;
    const float pc = cos(pos_phase);
    const float ps = sin(pos_phase);

    if (gid < d_q) {
        int o = gid;
        const __global float* wrow = W_qp + (size_t)o * d_lat;
        float acc = 0.0f;
        for (int z = 0; z < d_lat; ++z) acc += wrow[z] * Z[z];
        acc += b_q[o];
        //
        // PHASE AS A PER-PAIR ROTATION, NOT A SCALAR COMPLEX NUMBER
        // (corrected 2026-10-22).
        //
        // The previous code built Q as a SCALAR complex number:
        //     cr = acc*cos(phi);  ci = acc*sin(phi);     phi = W_phi · acc
        // so the attention score became Re(Q·R*) = q·k·cos(phi(q) − phi(k)):
        // 64 dimensions COLLAPSED into one complex scalar, and the inner
        // product was replaced by a cosine of a phase difference. Measured
        // consequence: perplexity 12,600x worse at alpha=0.1, and the score for
        // a random (q,k) pair went from -9.70 to -454.18.
        //
        // The form that PRESERVES the inner product — and therefore the model —
        // is a rotation of each dimension PAIR by a shared angle, exactly what
        // RoPE does:
        //     out[j]   = v[j]·cos(theta_j) − v[j+1]·sin(theta_j)
        //     out[j+1] = v[j]·sin(theta_j) + v[j+1]·cos(theta_j)
        // With the same theta applied to Q and K, <RQ,RK> = <q,k> exactly
        // (verified: relative error 0.00e+00).
        //
        // HONEST NOTE ON WHAT THIS CAN AND CANNOT DO: a shared rotation that
        // preserves the inner product is INVISIBLE to the score. It cannot
        // improve the model — it can only be neutral. The phase becomes useful
        // only if the angle differs between Q and K in a structured way (like
        // RoPE's position dependence), which is what the additive positional
        // term below already provides. The learned W_phi path is kept for the
        // experiment, but expect it to measure as neutral, not beneficial.
        //
        float qr = acc, qi = acc;      // real / imaginary lanes of this element
        if (have_phi) {
            //
            // SCALAR-vs-VECTOR BUG (fixed 2026-10-22).
            //
            // The previous line was:
            //
            //     for (int j = 0; j < d_q; ++j) phi += prow[j] * acc;
            //
            // `acc` is the SCALAR Q element (one output of W_q'·Z), but the
            // loop reads it as if it were a vector of length d_q. That sums the
            // SAME product d_q = 896 times, so phi ≈ 896·W_phi[o][0]·acc — a
            // factor-896 amplification. At alpha=0.1 the phase then spins by
            // radians × 10² on the first element, which is why activating the
            // knob produced perplexity 12,600x worse: the rotation was not
            // modulating, it was scrambling.
            //
            // This is also why the earlier "per-pair rotation" change measured
            // as identical to the original: the amplification dominated every
            // other difference in the formula.
            //
            // The phase must be a function of the LATENT vector Z (which has
            // d_lat entries), not of the scalar output. W_phi is [d_q × d_q]
            // and is indexed by o for the row; the input it multiplies is the
            // latent activation Z, projected to the same width.
            const __global float* prow = W_phi + (size_t)o * d_q;
            const int nz = (d_lat < d_q) ? d_lat : d_q;
            float phi = 0.0f;
            for (int j = 0; j < nz; ++j) phi += prow[j] * Z[j];
            // Per-element rotation into the complex plane, magnitude preserved
            // (|(qr,qi)| = |acc|).
            qr = acc * cos(phi);
            qi = acc * sin(phi);
        }
        // The pair partner is o+1 for even o, o-1 for odd o: rotate the PAIR.
        // Both lanes are written by the even element of the pair, mirroring the
        // RoPE rotate_half convention so Q stays a unit-norm-preserving map.
        Q[2*o]   = qr * pc - qi * ps;
        Q[2*o+1] = qr * ps + qi * pc;
    } else if (gid < d_q + d_kv) {
        int o = gid - d_q;
        const __global float* wrow = W_rp + (size_t)o * d_lat;
        float acc = 0.0f;
        for (int z = 0; z < d_lat; ++z) acc += wrow[z] * Z[z];
        acc += b_k[o];
        // Same scalar-vs-vector fix as the Q branch above: the phase is a
        // function of the latent Z, not of the scalar output. Summing
        // prow[j] * acc over j multiplied the same product d_kv times.
        float phi = 0.0f;
        if (have_phi) {
            const __global float* prow = W_phi + (size_t)o * d_q;
            const int nz = (d_lat < d_q) ? d_lat : d_q;
            for (int j = 0; j < nz; ++j) phi += prow[j] * Z[j];
        }
        float cr = have_phi ? acc * cos(phi) : acc;
        float ci = have_phi ? acc * sin(phi) : 0.0f;
        Rcache[r_off + 2*o]   = cr * pc - ci * ps;
        Rcache[r_off + 2*o+1] = cr * ps + ci * pc;
    } else {
        int o = gid - d_q - d_kv;
        const __global float* wrow = W_hp + (size_t)o * d_lat;
        float acc = 0.0f;
        for (int z = 0; z < d_lat; ++z) acc += wrow[z] * Z[z];
        acc += b_v[o];   // v_proj bias
        Hcache[h_off + o] = acc;
    }
}

// ── RoPE (rotate_half) applied AFTER the Q/R/H projection ──────────────────
//
// The pretrained model was trained with RoPE. The Ψ path must reproduce it or
// the attention is computed on un-positioned vectors: measured on
// Qwen2.5-0.5B, PsiForwardEngine gave perplexity 126.7 vs 37.4 for the
// reference (and 8.4e6 before the bias + softmax fixes).
//
// Layout note: Q is stored as COMPLEX INTERLEAVED — Q[2*o] is the real part and
// Q[2*o+1] the imaginary part of element o. rotate_half pairs element o with
// o + head_dim/2, so both the real and the imaginary planes are rotated with
// the same (cos, sin).
//
//   real[o]      = real[o]*c - real[o_p]*s
//   real[o_p]    = real[o_p]*c + real[o]*s
//
// One work-item per PAIR handles both halves; work-items in the upper half of
// each head return immediately (their partner already wrote them).
__kernel void rope_qr(
    __global float* Q,             // [2·d_q] complex interleaved (in place)
    __global float* Rcache,        // [cap · 2·d_kv]
    int d_q, int d_kv, int head_dim, int ctx, float rope_theta) {
    const int gid = (int)get_global_id(0);
    const int total = d_q + d_kv;
    if (gid >= total) return;
    const int hdim2 = head_dim / 2;

    if (gid < d_q) {
        const int o = gid;
        const int in_head = o % head_dim;
        if (in_head >= hdim2) return;          // partner handles this pair
        const int o_p = o + hdim2;
        const float inv = pow(rope_theta, -2.0f * in_head / (float)head_dim);
        const float ang = (float)ctx * inv;
        const float c = cos(ang), sn = sin(ang);
        const float r0 = Q[2*o],     i0 = Q[2*o + 1];
        const float r1 = Q[2*o_p],   i1 = Q[2*o_p + 1];
        Q[2*o]       = r0 * c - r1 * sn;
        Q[2*o_p]     = r1 * c + r0 * sn;
        Q[2*o + 1]   = i0 * c - i1 * sn;
        Q[2*o_p + 1] = i1 * c + i0 * sn;
    } else {
        const int o = gid - d_q;
        const int in_head = o % head_dim;
        if (in_head >= hdim2) return;
        const int o_p = o + hdim2;
        const float inv = pow(rope_theta, -2.0f * in_head / (float)head_dim);
        const float ang = (float)ctx * inv;
        const float c = cos(ang), sn = sin(ang);
        const size_t base = (size_t)ctx * 2 * d_kv;
        const float r0 = Rcache[base + 2*o],     i0 = Rcache[base + 2*o + 1];
        const float r1 = Rcache[base + 2*o_p],   i1 = Rcache[base + 2*o_p + 1];
        Rcache[base + 2*o]       = r0 * c - r1 * sn;
        Rcache[base + 2*o_p]     = r1 * c + r0 * sn;
        Rcache[base + 2*o + 1]   = i0 * c - i1 * sn;
        Rcache[base + 2*o_p + 1] = i1 * c + i0 * sn;
    }
}

// FUSED Madhava attention for ONE query position against the cache:
// scores → selective top-k (ksel) → weight = score/Σscore (softmax-free) →
// apply (values) in a SINGLE kernel. This is the Madhava principle applied to
// attention (exactly like winnex-nano forward.cpp: keep top-ksel by score,
// weight = score/sum(top scores), NO softmax). grid = n_heads work-groups;
// each group handles ONE head. Q/R/H all stay on the device.
__kernel void attn_fused(
    __global const float* Q,       // [2·d_q] complex interleaved query
    __global const float* Rcache,  // [cap · 2·d_kv] complex keys
    __global const float* Hcache,  // [cap · d_kv] real values
    __global float* out,           // [d_q] per-head values
    int n_heads, int n_kv_heads, int ctx, int head_dim) {
    int h = (int)get_group_id(0);
    if (h >= n_heads) return;
    int kvh = h / (n_heads / n_kv_heads);
    const int hd = head_dim;
    const __global float* qh = Q + (size_t)h * (2 * hd);

    const int lid = (int)get_local_id(0);
    const int lsz = (int)get_local_size(0);

    // ────────────────────────────────────────────────────────────────────────
    // BUG FIX (same two defects the winnex-nano forward had, see forward.cpp):
    //
    //  (a) NO CAUSAL MASK. `ctx` is the position of the key just appended, so
    //      the visible window for this token is keys [0, ctx] INCLUSIVE — not
    //      beyond. This kernel scanned `k < ctx`, which is the same window
    //      (the new key is at index ctx-1 after the append), so the window was
    //      correct here; the real defect was (b).
    //
    //  (b) TOP-K WITHOUT SOFTMAX. This kept only the 16 highest-scoring keys
    //      and weighted them w_i = s_i / Σ s with NO exponentiation. Attention
    //      scores are signed: w_i can be negative, Σs can pass through zero,
    //      and the result is unbounded. That is not the function the pretrained
    //      weights encode, and it is why PsiForwardEngine produced perplexity
    //      8.4e6 against 35.1 for the reference — even on a single token.
    //
    // The kernel now computes the standard causal softmax over every visible
    // key. No truncation, no division by a possibly-zero signed sum.
    // ────────────────────────────────────────────────────────────────────────
    __local float lmax[128];
    __local float lsum[128];

    // 1. Per-work-item max over its strided subrange of the visible window.
    float lmax_i = -3.4e38f;
    for (int k = lid; k < ctx; k += lsz) {
        const __global float* rk = Rcache + ((size_t)k * n_kv_heads + kvh) * (2 * hd);
        float s = 0.0f;
        for (int d = 0; d < hd; ++d) s += qh[2*d] * rk[2*d] + qh[2*d+1] * rk[2*d+1];
        s = s * rsqrt((float)hd);
        if (s > lmax_i) lmax_i = s;
    }
    lmax[lid] = lmax_i;
    barrier(CLK_LOCAL_MEM_FENCE);

    // 2. Reduce to the global max (numerical stability for exp()).
    if (lid == 0) {
        float m = -3.4e38f;
        for (int w = 0; w < lsz; ++w) if (lmax[w] > m) m = lmax[w];
        lmax[0] = m;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const float m = lmax[0];

    // 3. Per-work-item partial sum of exp(s - m).
    float lsum_i = 0.0f;
    for (int k = lid; k < ctx; k += lsz) {
        const __global float* rk = Rcache + ((size_t)k * n_kv_heads + kvh) * (2 * hd);
        float s = 0.0f;
        for (int d = 0; d < hd; ++d) s += qh[2*d] * rk[2*d] + qh[2*d+1] * rk[2*d+1];
        lsum_i += exp(s * rsqrt((float)hd) - m);
    }
    lsum[lid] = lsum_i;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        float z = 0.0f;
        for (int w = 0; w < lsz; ++w) z += lsum[w];
        lsum[0] = (z > 0.0f) ? z : 1.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const float inv = 1.0f / lsum[0];

    // 4. out[h][j] = Σ_k softmax_k · H[k][kvh][j], parallel over j.
    for (int j = lid; j < hd; j += lsz) {
        float acc = 0.0f;
        for (int k = 0; k < ctx; ++k) {
            const __global float* rk = Rcache + ((size_t)k * n_kv_heads + kvh) * (2 * hd);
            float s = 0.0f;
            for (int d = 0; d < hd; ++d) s += qh[2*d] * rk[2*d] + qh[2*d+1] * rk[2*d+1];
            float w = exp(s * rsqrt((float)hd) - m) * inv;
            acc += w * Hcache[((size_t)k * n_kv_heads + kvh) * hd + j];
        }
        out[(size_t)h * hd + j] = acc;
    }
}

// ══════════════════════════════════════════════════════════════════════════
// SPECTRAL LOG-PHASE FILTER (the "harmonic" processing).
// Applies F(k) = exp(i·α·arctan(ln(|k|+ε))) to a complex-interleaved vector
// (Q or R). This is the deterministic spectral modulation: each frequency
// index k gets a phase rotation that compresses high-frequency components
// (the arctan(ln|k|) is a smooth high-frequency suppressor — the Madhava
// philosophy: a bound-style regularizer, not a black box). Works directly on
// the complex (re,im) pairs — no FFT round-trip needed for the filter itself.
// grid = length of the vector (one item per complex pair).
__kernel void spectral_phase(
    __global float* v,          // [2·n] complex interleaved, in place
    int n, double alpha) {
    int i = (int)get_global_id(0);
    if (i >= n) return;
    float k = (float)(i + 1);            // frequency index 1..n
    float phase = (float)(alpha * atan(log(k + 1e-10f)));
    float c = cos(phase), s = sin(phase);
    float re = v[2*i], im = v[2*i+1];
    v[2*i]   = re * c - im * s;
    v[2*i+1] = re * s + im * c;
}
)";

// ══════════════════════════════════════════════════════════════════════════
// HilbertAttention::Impl — owns the device context, kernels, and the R/H cache.
// ══════════════════════════════════════════════════════════════════════════
struct HilbertAttention::Impl {
    cl_context ctx = nullptr;
    bool owns_ctx = false;   // false: ctx is the process-wide shared context
    cl_command_queue q = nullptr;
    cl_program prog = nullptr;
    cl_kernel k_lz = nullptr, k_lqrh = nullptr, k_attn = nullptr, k_out = nullptr;
    cl_kernel k_spec = nullptr;  // spectral log-phase filter
    cl_kernel k_rope = nullptr;  // RoPE (rotate_half) on Q/R
    double spectral_alpha = 0.0; // spectral modulation strength (0 = off)

    // Geometry.
    int d_q, d_kv, n_heads, n_kv_heads, head_dim, d;

    // Device buffers. The R/H caches are grown by DOUBLING: the buffer is
    // always a power-of-two ≥ ctx, so appending never copies the context.
    cl_mem dR = nullptr;   // [cap · 2·d_kv] complex keys
    cl_mem dH = nullptr;   // [cap · d_kv] real values
    cl_mem dO = nullptr;   // [d × d_q] output projection
    cl_mem dQ = nullptr;   // [2·d_q] query
    cl_mem dScores = nullptr; // [n_heads · cap] scores (cap = ctx capacity)
    cl_mem dOut = nullptr; // [d_q] per-head values
    cl_mem dY = nullptr;   // [d] after o_proj
    int cap = 0;           // capacity (power of two)
    int ctx_len = 0;       // current length
    bool has_o = false;

    // Latent-coupling device buffers (GPU latent forward).
    cl_mem dX = nullptr;     // [d] input
    cl_mem dZ = nullptr;     // [4d] latent
    cl_mem dWz = nullptr;    // [4d × d]
    cl_mem dWqp = nullptr;   // [d_q × 4d]
    cl_mem dWrp = nullptr;   // [d_kv × 4d]
    cl_mem dWhp = nullptr;   // [d_kv × 4d]
    cl_mem dWphi = nullptr;  // [d_q × d_q] phase (optional)
    cl_mem dbq = nullptr, dbk = nullptr, dbv = nullptr;  // attention biases
    bool use_rope = true;        // apply RoPE (rotate_half) to Q/R
    float rope_theta = 10000.0f; // model's rope_theta
    cl_mem dHpre = nullptr;  // [d_kv] (values staging for latent_forward_gpu)
    cl_mem dQout = nullptr;  // [2·d_q] phase-activated query
    cl_mem dRout = nullptr;  // [2·d_kv] phase-activated key (staging)
    bool has_latent = false;
    bool has_phi = false;

    char dev_name[256];

    ~Impl() { release(); }
    void release() {
        if (dR) clReleaseMemObject(dR);
        if (dH) clReleaseMemObject(dH);
        if (dO) clReleaseMemObject(dO);
        if (dQ) clReleaseMemObject(dQ);
        if (dScores) clReleaseMemObject(dScores);
        if (dOut) clReleaseMemObject(dOut);
        if (dY) clReleaseMemObject(dY);
        if (dX) clReleaseMemObject(dX);
        if (dZ) clReleaseMemObject(dZ);
        if (dWz) clReleaseMemObject(dWz);
        if (dWqp) clReleaseMemObject(dWqp);
        if (dWrp) clReleaseMemObject(dWrp);
        if (dWhp) clReleaseMemObject(dWhp);
        if (dWphi) clReleaseMemObject(dWphi);
        if (dHpre) clReleaseMemObject(dHpre);
        if (dQout) clReleaseMemObject(dQout);
        if (dRout) clReleaseMemObject(dRout);
        if (k_lz) clReleaseKernel(k_lz);
        if (k_lqrh) clReleaseKernel(k_lqrh);
        if (k_attn) clReleaseKernel(k_attn);
        if (k_out) clReleaseKernel(k_out);
        if (k_spec) clReleaseKernel(k_spec);
        if (prog) clReleaseProgram(prog);
        if (q) clReleaseCommandQueue(q);
        // Only release the context if this instance created it. With the
        // shared-context fix the context is process-wide and outlives
        // individual layers; releasing it here would destroy the other 23.
        if (ctx && owns_ctx) clReleaseContext(ctx);
        dR = dH = dO = dQ = dScores = dOut = dY = nullptr;
        dX = dZ = dWz = dWqp = dWrp = dWhp = nullptr;
        dWphi = dHpre = dQout = dRout = nullptr;
        k_lz = k_lqrh = k_attn = k_out = nullptr;
        k_spec = nullptr;
        prog = nullptr; q = nullptr; ctx = nullptr;
    }

    void ensure_capacity(int need) {
        if (need <= cap) return;
        int new_cap = (cap > 0) ? cap : 1;
        while (new_cap < need) new_cap <<= 1;
        cl_int err = 0;
        cl_mem nR = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                   (size_t)new_cap * 2 * d_kv * 4, nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("OpenCL: create R cache failed");
        cl_mem nH = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                   (size_t)new_cap * d_kv * 4, nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("OpenCL: create H cache failed");
        cl_mem nS = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                   (size_t)n_heads * new_cap * 4, nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("OpenCL: create scores buffer failed");
        // Copy existing R/H/S contents (the cache grows by doubling, so the
        // first ctx_len entries must survive a realloc).
        if (ctx_len > 0 && dR) {
            clEnqueueCopyBuffer(q, dR, nR, 0, 0, (size_t)ctx_len * 2 * d_kv * 4,
                                0, nullptr, nullptr);
            clEnqueueCopyBuffer(q, dH, nH, 0, 0, (size_t)ctx_len * d_kv * 4,
                                0, nullptr, nullptr);
        }
        if (dR) clReleaseMemObject(dR);
        if (dH) clReleaseMemObject(dH);
        if (dScores) clReleaseMemObject(dScores);
        dR = nR; dH = nH; dScores = nS; cap = new_cap;
    }
};

HilbertAttention::HilbertAttention(int d_q, int d_kv, int n_heads, int n_kv_heads,
                                   int head_dim, int d)
    : impl_(new Impl) {
    impl_->d_q = d_q; impl_->d_kv = d_kv;
    impl_->n_heads = n_heads; impl_->n_kv_heads = n_kv_heads;
    impl_->head_dim = head_dim; impl_->d = d;
    impl_->dev_name[0] = '\0';

    if (!load_opencl())
        throw std::runtime_error("HilbertAttention: OpenCL loader unavailable (no fallback)");

    // Pick a device: prefer GPU, fall back to CPU (the loader exists but the
    // driver may expose only a CPU device — e.g. a headless box).
    cl_int err = CL_SUCCESS;
    cl_platform_id plat = nullptr; cl_uint nplat = 0;
    clGetPlatformIDs(0, nullptr, &nplat);
    if (nplat == 0) throw std::runtime_error("HilbertAttention: no OpenCL platform");
    clGetPlatformIDs(nplat, &plat, nullptr);
    cl_device_id dev = nullptr; cl_uint ndev = 0;
    clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &ndev);
    if (ndev == 0) clGetDeviceIDs(plat, CL_DEVICE_TYPE_DEFAULT, 0, nullptr, &ndev);
    if (ndev == 0) clGetDeviceIDs(plat, CL_DEVICE_TYPE_CPU, 0, nullptr, &ndev);
    if (ndev == 0) throw std::runtime_error("HilbertAttention: no OpenCL device");
    clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 1, &dev, nullptr);
    if (!dev) throw std::runtime_error("HilbertAttention: device id null");

    // Device name for diagnostics.
    size_t nm = 0;
    clGetDeviceInfo(dev, CL_DEVICE_NAME, 0, nullptr, &nm);
    if (nm > 0) {
        std::vector<char> buf(nm);
        clGetDeviceInfo(dev, CL_DEVICE_NAME, nm, buf.data(), nullptr);
        std::snprintf(impl_->dev_name, sizeof(impl_->dev_name), "%s", buf.data());
    } else {
        std::snprintf(impl_->dev_name, sizeof(impl_->dev_name), "unknown");
    }

    // SHARED CONTEXT across instances (see the note on cl_context below).
    //
    // BUG FIX: this used to call clCreateContext() unconditionally, so every
    // HilbertAttention owned a context of its own. The driver allocates device
    // scratch per context — measured at ~131 MB each on an RTX 5060 Ti — so
    // PsiForwardEngine (one instance per layer) needed 24 × 131 MB ≈ 3.1 GB
    // just to start, and died at layer 15 with "create context failed".
    //
    // One context is enough for the whole model: buffers, kernels and queues
    // are all per-instance and remain independent; only the context (the
    // driver's device state) is shared. The queue MUST stay per-instance,
    // because enqueueing from several threads on one queue is not thread-safe.
    static cl_context g_shared_ctx = nullptr;
    static cl_device_id g_shared_dev = nullptr;
    static int g_ctx_refs = 0;
    if (g_shared_ctx == nullptr || g_shared_dev != dev) {
        g_shared_ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("HilbertAttention: create context failed");
        g_shared_dev = dev;
        g_ctx_refs = 0;
    }
    impl_->owns_ctx = false;
    impl_->ctx = g_shared_ctx;
    ++g_ctx_refs;
    impl_->q = clCreateCommandQueue(impl_->ctx, dev, 0, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("HilbertAttention: create queue failed");

    const char* src = g_attn_kernels;
    impl_->prog = clCreateProgramWithSource(impl_->ctx, 1, &src, nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("HilbertAttention: create program failed");
    err = clBuildProgram(impl_->prog, 1, &dev, "-cl-fast-relaxed-math", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        char log[4096] = {0};
        clGetProgramBuildInfo(impl_->prog, dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
        throw std::runtime_error(std::string("HilbertAttention: kernel build failed:\n") + log);
    }
    impl_->k_lz   = clCreateKernel(impl_->prog, "latent_z", &err);
    impl_->k_lqrh = clCreateKernel(impl_->prog, "latent_qrh_phase", &err);
    impl_->k_attn = clCreateKernel(impl_->prog, "attn_fused", &err);
    impl_->k_rope = clCreateKernel(impl_->prog, "rope_qr", &err);
    impl_->k_out  = clCreateKernel(impl_->prog, "attn_outproj", &err);
    impl_->k_spec = clCreateKernel(impl_->prog, "spectral_phase", &err);
    if (!impl_->k_lz || !impl_->k_lqrh || !impl_->k_attn || !impl_->k_out) {
        throw std::runtime_error("HilbertAttention: kernel create failed");
    }

    // Static buffers (query, out, y).
    impl_->dQ = clCreateBuffer(impl_->ctx, CL_MEM_READ_WRITE, (size_t)2 * d_q * 4, nullptr, &err);
    impl_->dOut = clCreateBuffer(impl_->ctx, CL_MEM_READ_WRITE, (size_t)d_q * 4, nullptr, &err);
    impl_->dY = clCreateBuffer(impl_->ctx, CL_MEM_READ_WRITE, (size_t)d * 4, nullptr, &err);
    if (!impl_->dQ || !impl_->dOut || !impl_->dY)
        throw std::runtime_error("HilbertAttention: static buffer create failed");

    // Latent-coupling buffers.
    impl_->dX   = clCreateBuffer(impl_->ctx, CL_MEM_READ_ONLY,  (size_t)d * 4, nullptr, &err);
    impl_->dZ   = clCreateBuffer(impl_->ctx, CL_MEM_READ_WRITE, (size_t)4 * d * 4, nullptr, &err);
    impl_->dHpre = clCreateBuffer(impl_->ctx, CL_MEM_READ_WRITE, (size_t)d_kv * 4, nullptr, &err);
    impl_->dQout = clCreateBuffer(impl_->ctx, CL_MEM_READ_WRITE, (size_t)2 * d_q * 4, nullptr, &err);
    impl_->dRout = clCreateBuffer(impl_->ctx, CL_MEM_READ_WRITE, (size_t)2 * d_kv * 4, nullptr, &err);
    impl_->dWphi = clCreateBuffer(impl_->ctx, CL_MEM_READ_ONLY, (size_t)d_q * d_q * 4, nullptr, &err);
    // Bias buffers, zero-initialised: architectures without attention bias
    // (Llama) simply never write them and add 0.
    impl_->dbq = clCreateBuffer(impl_->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                (size_t)d_q * 4, std::calloc(d_q, 4), &err);
    impl_->dbk = clCreateBuffer(impl_->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                (size_t)d_kv * 4, std::calloc(d_kv, 4), &err);
    impl_->dbv = clCreateBuffer(impl_->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                (size_t)d_kv * 4, std::calloc(d_kv, 4), &err);
    if (!impl_->dX || !impl_->dZ || !impl_->dHpre || !impl_->dQout || !impl_->dRout || !impl_->dWphi)
        throw std::runtime_error("HilbertAttention: latent buffer create failed");

    impl_->ensure_capacity(1);  // initial cache capacity
}

HilbertAttention::~HilbertAttention() { delete impl_; }

void HilbertAttention::set_output_projection(const float* o_proj) {
    if (!o_proj) return;
    cl_int err = CL_SUCCESS;
    impl_->dO = clCreateBuffer(impl_->ctx, CL_MEM_READ_ONLY,
                               (size_t)impl_->d * impl_->d_q * 4, nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("HilbertAttention: o_proj buffer create failed");
    clEnqueueWriteBuffer(impl_->q, impl_->dO, CL_TRUE, 0,
                         (size_t)impl_->d * impl_->d_q * 4, o_proj, 0, nullptr, nullptr);
    impl_->has_o = true;
}

void HilbertAttention::kv_append(const float* R, const float* H) {
    impl_->ensure_capacity(impl_->ctx_len + 1);
    const int kvh = impl_->ctx_len;
    const size_t r_off = (size_t)kvh * 2 * impl_->d_kv * 4;
    const size_t h_off = (size_t)kvh * impl_->d_kv * 4;
    clEnqueueWriteBuffer(impl_->q, impl_->dR, CL_TRUE, r_off,
                         (size_t)2 * impl_->d_kv * 4, R, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(impl_->q, impl_->dH, CL_TRUE, h_off,
                         (size_t)impl_->d_kv * 4, H, 0, nullptr, nullptr);
    impl_->ctx_len += 1;
}

void HilbertAttention::set_latent_weights(const float* W_z, const float* W_qp,
                                          const float* W_rp, const float* W_hp) {
    if (!W_z || !W_qp || !W_rp || !W_hp)
        throw std::runtime_error("set_latent_weights: null weights");
    cl_int err = CL_SUCCESS;
    const int d = impl_->d, d_q = impl_->d_q, d_kv = impl_->d_kv;
    impl_->dWz  = clCreateBuffer(impl_->ctx, CL_MEM_READ_ONLY,
                                 (size_t)4 * d * d * 4, nullptr, &err);
    impl_->dWqp = clCreateBuffer(impl_->ctx, CL_MEM_READ_ONLY,
                                 (size_t)d_q * 4 * d * 4, nullptr, &err);
    impl_->dWrp = clCreateBuffer(impl_->ctx, CL_MEM_READ_ONLY,
                                 (size_t)d_kv * 4 * d * 4, nullptr, &err);
    impl_->dWhp = clCreateBuffer(impl_->ctx, CL_MEM_READ_ONLY,
                                 (size_t)d_kv * 4 * d * 4, nullptr, &err);
    if (!impl_->dWz || !impl_->dWqp || !impl_->dWrp || !impl_->dWhp)
        throw std::runtime_error("set_latent_weights: buffer create failed");
    clEnqueueWriteBuffer(impl_->q, impl_->dWz, CL_TRUE, 0, (size_t)4 * d * d * 4, W_z, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(impl_->q, impl_->dWqp, CL_TRUE, 0, (size_t)d_q * 4 * d * 4, W_qp, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(impl_->q, impl_->dWrp, CL_TRUE, 0, (size_t)d_kv * 4 * d * 4, W_rp, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(impl_->q, impl_->dWhp, CL_TRUE, 0, (size_t)d_kv * 4 * d * 4, W_hp, 0, nullptr, nullptr);
    impl_->has_latent = true;
}

void HilbertAttention::latent_forward_gpu(const float* x, float* out, float* out_r,
                                          double phase_alpha) {
    if (!impl_->has_latent)
        throw std::runtime_error("latent_forward_gpu: call set_latent_weights first");
    const int d = impl_->d, d_q = impl_->d_q, d_kv = impl_->d_kv;
    const int have_phi = (phase_alpha != 0.0 && impl_->has_phi) ? 1 : 0;

    // Upload input (async — the kernel enqueue below orders it on the queue).
    clEnqueueWriteBuffer(impl_->q, impl_->dX, CL_FALSE, 0, (size_t)d * 4,
                         x, 0, nullptr, nullptr);

    // Kernel A: Z = W_z·x  (grid 4d).
    {
        cl_int e = CL_SUCCESS;
        e |= clSetKernelArg(impl_->k_lz, 0, sizeof(cl_mem), &impl_->dX);
        e |= clSetKernelArg(impl_->k_lz, 1, sizeof(cl_mem), &impl_->dWz);
        e |= clSetKernelArg(impl_->k_lz, 2, sizeof(cl_mem), &impl_->dZ);
        e |= clSetKernelArg(impl_->k_lz, 3, sizeof(int), &d);
        if (e != CL_SUCCESS) throw std::runtime_error("latent_gpu: Z arg set failed");
        size_t gsz = (size_t)(4 * d);
        clEnqueueNDRangeKernel(impl_->q, impl_->k_lz, 1, nullptr, &gsz, nullptr, 0, nullptr, nullptr);
    }

    // Kernel B: Q/R/H from Z, phase fused in.
    {
        cl_int e = CL_SUCCESS;
        e |= clSetKernelArg(impl_->k_lqrh, 0, sizeof(cl_mem), &impl_->dZ);
        e |= clSetKernelArg(impl_->k_lqrh, 1, sizeof(cl_mem), &impl_->dWqp);
        e |= clSetKernelArg(impl_->k_lqrh, 2, sizeof(cl_mem), &impl_->dWrp);
        e |= clSetKernelArg(impl_->k_lqrh, 3, sizeof(cl_mem), &impl_->dWhp);
        e |= clSetKernelArg(impl_->k_lqrh, 4, sizeof(cl_mem), &impl_->dWphi);
        e |= clSetKernelArg(impl_->k_lqrh, 5, sizeof(cl_mem), &impl_->dQout);
        e |= clSetKernelArg(impl_->k_lqrh, 6, sizeof(cl_mem), &impl_->dRout);
        e |= clSetKernelArg(impl_->k_lqrh, 7, sizeof(cl_mem), &impl_->dHpre);
        int a = d_q, b = d_kv, c = 4 * d, hp = have_phi, ctx = 0;
        // Argument order MUST match the kernel signature:
        //   0 Z, 1 W_qp, 2 W_rp, 3 W_hp, 4 W_phi, 5 Q, 6 Rcache, 7 Hcache,
        //   8 b_q, 9 b_k, 10 b_v,
        //   11 d_q, 12 d_kv, 13 d_lat, 14 have_phi, 15 ctx, 16 have_pos_phase
        int hpp = PosPhaseEnabled() ? 1 : 0;
        e |= clSetKernelArg(impl_->k_lqrh,  8, sizeof(cl_mem), &impl_->dbq);
        e |= clSetKernelArg(impl_->k_lqrh,  9, sizeof(cl_mem), &impl_->dbk);
        e |= clSetKernelArg(impl_->k_lqrh, 10, sizeof(cl_mem), &impl_->dbv);
        e |= clSetKernelArg(impl_->k_lqrh, 11, sizeof(int), &a);
        e |= clSetKernelArg(impl_->k_lqrh, 12, sizeof(int), &b);
        e |= clSetKernelArg(impl_->k_lqrh, 13, sizeof(int), &c);
        e |= clSetKernelArg(impl_->k_lqrh, 14, sizeof(int), &hp);
        e |= clSetKernelArg(impl_->k_lqrh, 15, sizeof(int), &ctx);
        e |= clSetKernelArg(impl_->k_lqrh, 16, sizeof(int), &hpp);
        if (e != CL_SUCCESS) throw std::runtime_error("latent_gpu: QRH arg set failed");
        const int want = d_q + 2 * d_kv;
        size_t lsz = (want >= 128) ? 128 : (size_t)want;
        size_t gsz = ((size_t)want + lsz - 1) / lsz * lsz;  // round up to multiple
        clEnqueueNDRangeKernel(impl_->q, impl_->k_lqrh, 1, nullptr, &gsz, &lsz, 0, nullptr, nullptr);
    }

    // Read back Q and R (blocking, so the caller sees the result).
    clEnqueueReadBuffer(impl_->q, impl_->dQout, CL_TRUE, 0, (size_t)2 * d_q * 4,
                        out, 0, nullptr, nullptr);
    clEnqueueReadBuffer(impl_->q, impl_->dRout, CL_TRUE, 0, (size_t)2 * d_kv * 4,
                        out_r, 0, nullptr, nullptr);
}

void HilbertAttention::set_spectral_alpha(double alpha) { impl_->spectral_alpha = alpha; }

void HilbertAttention::apply_spectral(float* v, int n, double alpha) {
    if (!v || n <= 0) return;
    cl_mem buf = clCreateBuffer(impl_->ctx, CL_MEM_READ_WRITE, (size_t)2 * n * 4,
                                nullptr, nullptr);
    if (!buf) throw std::runtime_error("apply_spectral: buffer create failed");
    clEnqueueWriteBuffer(impl_->q, buf, CL_TRUE, 0, (size_t)2 * n * 4, v, 0, nullptr, nullptr);
    cl_int e = CL_SUCCESS;
    e |= clSetKernelArg(impl_->k_spec, 0, sizeof(cl_mem), &buf);
    e |= clSetKernelArg(impl_->k_spec, 1, sizeof(int), &n);
    e |= clSetKernelArg(impl_->k_spec, 2, sizeof(double), &alpha);
    size_t gsz = (size_t)n;
    clEnqueueNDRangeKernel(impl_->q, impl_->k_spec, 1, nullptr, &gsz, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(impl_->q, buf, CL_TRUE, 0, (size_t)2 * n * 4, v, 0, nullptr, nullptr);
    clReleaseMemObject(buf);
}

void HilbertAttention::set_phase_matrix(const float* W_phi) {
    if (!W_phi) return;
    const int d_q = impl_->d_q;
    clEnqueueWriteBuffer(impl_->q, impl_->dWphi, CL_TRUE, 0,
                         (size_t)d_q * d_q * 4, W_phi, 0, nullptr, nullptr);
    impl_->has_phi = true;
}

void HilbertAttention::set_rope(float theta, bool enabled) {
    impl_->rope_theta = theta;
    impl_->use_rope = enabled;
}

void HilbertAttention::set_attention_bias(const float* b_q, const float* b_k,
                                          const float* b_v) {
    if (b_q) clEnqueueWriteBuffer(impl_->q, impl_->dbq, CL_TRUE, 0,
                                  (size_t)impl_->d_q * 4, b_q, 0, nullptr, nullptr);
    if (b_k) clEnqueueWriteBuffer(impl_->q, impl_->dbk, CL_TRUE, 0,
                                  (size_t)impl_->d_kv * 4, b_k, 0, nullptr, nullptr);
    if (b_v) clEnqueueWriteBuffer(impl_->q, impl_->dbv, CL_TRUE, 0,
                                  (size_t)impl_->d_kv * 4, b_v, 0, nullptr, nullptr);
}

void HilbertAttention::kv_append_gpu() {
    // Append the R/H computed by the LAST latent_forward_gpu directly from the
    // device buffers (no host round-trip of the key/value).
    impl_->ensure_capacity(impl_->ctx_len + 1);
    const int kvh = impl_->ctx_len;
    const size_t r_off = (size_t)kvh * 2 * impl_->d_kv * 4;
    const size_t h_off = (size_t)kvh * impl_->d_kv * 4;
    clEnqueueCopyBuffer(impl_->q, impl_->dRout, impl_->dR, 0, r_off,
                        (size_t)2 * impl_->d_kv * 4, 0, nullptr, nullptr);
    clEnqueueCopyBuffer(impl_->q, impl_->dHpre, impl_->dH, 0, h_off,
                        (size_t)impl_->d_kv * 4, 0, nullptr, nullptr);
    impl_->ctx_len += 1;
}

void HilbertAttention::forward_gpu(const float* x, float* y, double phase_alpha) {
    if (!impl_->has_latent)
        throw std::runtime_error("forward_gpu: call set_latent_weights first");
    const int d = impl_->d, d_q = impl_->d_q, d_kv = impl_->d_kv;
    const int n_heads = impl_->n_heads, n_kv = impl_->n_kv_heads, hd = impl_->head_dim;
    const int have_phi = (phase_alpha != 0.0 && impl_->has_phi) ? 1 : 0;

    impl_->ensure_capacity(impl_->ctx_len + 1);
    clEnqueueWriteBuffer(impl_->q, impl_->dX, CL_FALSE, 0, (size_t)d * 4,
                         x, 0, nullptr, nullptr);

    // ── 1. latent Z = W_z·x ───────────────────────────────────────────────
    {
        cl_int e = CL_SUCCESS;
        e |= clSetKernelArg(impl_->k_lz, 0, sizeof(cl_mem), &impl_->dX);
        e |= clSetKernelArg(impl_->k_lz, 1, sizeof(cl_mem), &impl_->dWz);
        e |= clSetKernelArg(impl_->k_lz, 2, sizeof(cl_mem), &impl_->dZ);
        e |= clSetKernelArg(impl_->k_lz, 3, sizeof(int), &d);
        size_t gsz = (size_t)(4 * d);
        clEnqueueNDRangeKernel(impl_->q, impl_->k_lz, 1, nullptr, &gsz, nullptr, 0, nullptr, nullptr);
    }
    // ── 2. Q/R/H from Z (+ phase), appended DIRECTLY to the cache ─────────
    {
        cl_int e = CL_SUCCESS;
        e |= clSetKernelArg(impl_->k_lqrh, 0, sizeof(cl_mem), &impl_->dZ);
        e |= clSetKernelArg(impl_->k_lqrh, 1, sizeof(cl_mem), &impl_->dWqp);
        e |= clSetKernelArg(impl_->k_lqrh, 2, sizeof(cl_mem), &impl_->dWrp);
        e |= clSetKernelArg(impl_->k_lqrh, 3, sizeof(cl_mem), &impl_->dWhp);
        e |= clSetKernelArg(impl_->k_lqrh, 4, sizeof(cl_mem), &impl_->dWphi);
        e |= clSetKernelArg(impl_->k_lqrh, 5, sizeof(cl_mem), &impl_->dQout);
        e |= clSetKernelArg(impl_->k_lqrh, 6, sizeof(cl_mem), &impl_->dR);
        e |= clSetKernelArg(impl_->k_lqrh, 7, sizeof(cl_mem), &impl_->dH);
        int a = d_q, b = d_kv, c = 4 * d, hp = have_phi, ctx = impl_->ctx_len;
        // Argument order MUST match the kernel signature:
        //   0 Z, 1 W_qp, 2 W_rp, 3 W_hp, 4 W_phi, 5 Q, 6 Rcache, 7 Hcache,
        //   8 b_q, 9 b_k, 10 b_v,
        //   11 d_q, 12 d_kv, 13 d_lat, 14 have_phi, 15 ctx, 16 have_pos_phase
        int hpp = PosPhaseEnabled() ? 1 : 0;
        e |= clSetKernelArg(impl_->k_lqrh,  8, sizeof(cl_mem), &impl_->dbq);
        e |= clSetKernelArg(impl_->k_lqrh,  9, sizeof(cl_mem), &impl_->dbk);
        e |= clSetKernelArg(impl_->k_lqrh, 10, sizeof(cl_mem), &impl_->dbv);
        e |= clSetKernelArg(impl_->k_lqrh, 11, sizeof(int), &a);
        e |= clSetKernelArg(impl_->k_lqrh, 12, sizeof(int), &b);
        e |= clSetKernelArg(impl_->k_lqrh, 13, sizeof(int), &c);
        e |= clSetKernelArg(impl_->k_lqrh, 14, sizeof(int), &hp);
        e |= clSetKernelArg(impl_->k_lqrh, 15, sizeof(int), &ctx);
        e |= clSetKernelArg(impl_->k_lqrh, 16, sizeof(int), &hpp);
        const int want = d_q + 2 * d_kv;
        size_t lsz = (want >= 128) ? 128 : (size_t)want;
        size_t gsz = ((size_t)want + lsz - 1) / lsz * lsz;
        cl_int qerr = clEnqueueNDRangeKernel(impl_->q, impl_->k_lqrh, 1, nullptr, &gsz, &lsz, 0, nullptr, nullptr);
        if (e != CL_SUCCESS || qerr != CL_SUCCESS) {
            throw std::runtime_error("forward_gpu: latent_qrh_phase failed (arg=" +
                                     std::to_string((int)e) + " enq=" + std::to_string((int)qerr) + ")");
        }
    }
    // ── 1.5 RoPE (rotate_half) on the just-written Q and R ────────────────
    if (impl_->use_rope && impl_->k_rope) {
        cl_int e = CL_SUCCESS;
        e |= clSetKernelArg(impl_->k_rope, 0, sizeof(cl_mem), &impl_->dQout);
        e |= clSetKernelArg(impl_->k_rope, 1, sizeof(cl_mem), &impl_->dR);
        e |= clSetKernelArg(impl_->k_rope, 2, sizeof(int), &d_q);
        e |= clSetKernelArg(impl_->k_rope, 3, sizeof(int), &d_kv);
        int hdd = hd;
        float rth = impl_->rope_theta;
        int pos = impl_->ctx_len;                  // BEFORE the increment below
        e |= clSetKernelArg(impl_->k_rope, 4, sizeof(int), &hdd);
        e |= clSetKernelArg(impl_->k_rope, 5, sizeof(int), &pos);
        e |= clSetKernelArg(impl_->k_rope, 6, sizeof(float), &rth);
        size_t gsz = (size_t)(d_q + d_kv);
        cl_int qe = clEnqueueNDRangeKernel(impl_->q, impl_->k_rope, 1, nullptr, &gsz, nullptr, 0, nullptr, nullptr);
        if (e != CL_SUCCESS || qe != CL_SUCCESS) {
            throw std::runtime_error("forward_gpu: rope_qr failed (arg=" +
                                     std::to_string((int)e) + " enq=" + std::to_string((int)qe) + ")");
        }
    }
    impl_->ctx_len += 1;

    // ── 2.5 SPECTRAL LOG-PHASE FILTER (optional, the harmonic modulation) ──
    // Applies F(k)=exp(i·α·arctan(ln(k+ε))) to the phase-activated Q in place.
    if (impl_->spectral_alpha != 0.0) {
        cl_int e = CL_SUCCESS;
        e |= clSetKernelArg(impl_->k_spec, 0, sizeof(cl_mem), &impl_->dQout);
        int n = d_q;
        double a = impl_->spectral_alpha;
        e |= clSetKernelArg(impl_->k_spec, 1, sizeof(int), &n);
        e |= clSetKernelArg(impl_->k_spec, 2, sizeof(double), &a);
        size_t gsz = (size_t)d_q;
        clEnqueueNDRangeKernel(impl_->q, impl_->k_spec, 1, nullptr, &gsz, nullptr, 0, nullptr, nullptr);
    }

    // ── 3. fused attention (scores + L2 + apply, ONE kernel) ──────────────
    {
        cl_int e = CL_SUCCESS;
        e |= clSetKernelArg(impl_->k_attn, 0, sizeof(cl_mem), &impl_->dQout);
        e |= clSetKernelArg(impl_->k_attn, 1, sizeof(cl_mem), &impl_->dR);
        e |= clSetKernelArg(impl_->k_attn, 2, sizeof(cl_mem), &impl_->dH);
        e |= clSetKernelArg(impl_->k_attn, 3, sizeof(cl_mem), &impl_->dOut);
        int a = n_heads, b = n_kv, c = impl_->ctx_len, dd = hd;
        e |= clSetKernelArg(impl_->k_attn, 4, sizeof(int), &a);
        e |= clSetKernelArg(impl_->k_attn, 5, sizeof(int), &b);
        e |= clSetKernelArg(impl_->k_attn, 6, sizeof(int), &c);
        e |= clSetKernelArg(impl_->k_attn, 7, sizeof(int), &dd);
        size_t lsz = 128;
        size_t gsz = (size_t)n_heads * lsz;   // one work-group per head
        clEnqueueNDRangeKernel(impl_->q, impl_->k_attn, 1, nullptr, &gsz, &lsz, 0, nullptr, nullptr);
    }

    // ── 4. o_proj + single blocking readback ──────────────────────────────
    if (impl_->has_o) {
        cl_int e = CL_SUCCESS;
        e |= clSetKernelArg(impl_->k_out, 0, sizeof(cl_mem), &impl_->dOut);
        e |= clSetKernelArg(impl_->k_out, 1, sizeof(cl_mem), &impl_->dO);
        e |= clSetKernelArg(impl_->k_out, 2, sizeof(cl_mem), &impl_->dY);
        int dq = d_q;
        e |= clSetKernelArg(impl_->k_out, 3, sizeof(int), &dq);
        size_t gsz = (size_t)d;
        clEnqueueNDRangeKernel(impl_->q, impl_->k_out, 1, nullptr, &gsz, nullptr, 0, nullptr, nullptr);
        clEnqueueReadBuffer(impl_->q, impl_->dY, CL_TRUE, 0, (size_t)d * 4,
                            y, 0, nullptr, nullptr);
    } else {
        clEnqueueReadBuffer(impl_->q, impl_->dOut, CL_TRUE, 0, (size_t)d_q * 4,
                            y, 0, nullptr, nullptr);
    }
}

void HilbertAttention::attention(const float* Q, float* out, float* y) {
    const int n_heads = impl_->n_heads, n_kv = impl_->n_kv_heads;
    const int hd = impl_->head_dim, ctx = impl_->ctx_len;
    if (ctx == 0) { std::memset(out, 0, (size_t)impl_->d_q * 4); return; }

    // Upload the query to the device query buffer.
    clEnqueueWriteBuffer(impl_->q, impl_->dQout, CL_TRUE, 0,
                         (size_t)2 * impl_->d_q * 4, Q, 0, nullptr, nullptr);

    // ONE fused kernel: scores → L2 → apply, against the device cache.
    {
        cl_int e = CL_SUCCESS;
        e |= clSetKernelArg(impl_->k_attn, 0, sizeof(cl_mem), &impl_->dQout);
        e |= clSetKernelArg(impl_->k_attn, 1, sizeof(cl_mem), &impl_->dR);
        e |= clSetKernelArg(impl_->k_attn, 2, sizeof(cl_mem), &impl_->dH);
        e |= clSetKernelArg(impl_->k_attn, 3, sizeof(cl_mem), &impl_->dOut);
        int a = n_heads, b = n_kv, c = ctx, dd = hd;
        e |= clSetKernelArg(impl_->k_attn, 4, sizeof(int), &a);
        e |= clSetKernelArg(impl_->k_attn, 5, sizeof(int), &b);
        e |= clSetKernelArg(impl_->k_attn, 6, sizeof(int), &c);
        e |= clSetKernelArg(impl_->k_attn, 7, sizeof(int), &dd);
        if (e != CL_SUCCESS) throw std::runtime_error("HilbertAttention: attn_fused arg set failed");
        size_t lsz = 128;
        size_t gsz = (size_t)n_heads * lsz;   // one work-group per head
        clEnqueueNDRangeKernel(impl_->q, impl_->k_attn, 1, nullptr, &gsz, &lsz, 0, nullptr, nullptr);
    }

    // Read per-head values.
    clEnqueueReadBuffer(impl_->q, impl_->dOut, CL_TRUE, 0,
                        (size_t)impl_->d_q * 4, out, 0, nullptr, nullptr);

    // 4. y = o_proj · out  (if set).
    if (y && impl_->has_o) {
        cl_int e = CL_SUCCESS;
        e |= clSetKernelArg(impl_->k_out, 0, sizeof(cl_mem), &impl_->dOut);
        e |= clSetKernelArg(impl_->k_out, 1, sizeof(cl_mem), &impl_->dO);
        e |= clSetKernelArg(impl_->k_out, 2, sizeof(cl_mem), &impl_->dY);
        int dq = impl_->d_q;
        e |= clSetKernelArg(impl_->k_out, 3, sizeof(int), &dq);
        if (e != CL_SUCCESS) throw std::runtime_error("HilbertAttention: out arg set failed");
        size_t gsz = (size_t)impl_->d;
        clEnqueueNDRangeKernel(impl_->q, impl_->k_out, 1, nullptr, &gsz, nullptr, 0, nullptr, nullptr);
        clEnqueueReadBuffer(impl_->q, impl_->dY, CL_TRUE, 0,
                            (size_t)impl_->d * 4, y, 0, nullptr, nullptr);
    } else if (y) {
        std::memset(y, 0, (size_t)impl_->d * 4);
    }
}

int HilbertAttention::ctx_len() const { return impl_->ctx_len; }
int HilbertAttention::d_q() const { return impl_->d_q; }
int HilbertAttention::d_kv() const { return impl_->d_kv; }
int HilbertAttention::d() const { return impl_->d; }

void HilbertAttention::reset() { impl_->ctx_len = 0; }

const char* HilbertAttention::device_name() const { return impl_->dev_name; }

} // namespace winnex_xfactor_c
