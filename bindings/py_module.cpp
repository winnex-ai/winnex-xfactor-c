// py_module.cpp — Python bindings for winnex-xfactor-c (pybind11).
//
// Exposes the native Ψ attention core: weight derivation (adaptivity),
// phase activation, latent coupling forward, and the phase-modulated score.
//
//   import winnex_xfactor_c
//   W_z, W_qp, W_rp, W_hp = winnex_xfactor_c.derive_from_qkv(W_q, W_k, W_v)
#include "winnex_xfactor_c/derive_weights.hpp"
#include "winnex_xfactor_c/latent_coupling.hpp"
#include "winnex_xfactor_c/attention_forward.hpp"
#include "winnex_xfactor_c/psi_forward.hpp"

#include <cstring>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using winnex_xfactor_c::AttentionConfig;
using winnex_xfactor_c::DerivedWeights;
using winnex_xfactor_c::HilbertAttention;
using winnex_xfactor_c::PsiForwardEngine;
using winnex_xfactor_c::derive_weights;
using winnex_xfactor_c::latent_forward;
using winnex_xfactor_c::phase_factors;
using winnex_xfactor_c::preservation_error;
using winnex_xfactor_c::score_phase;

PYBIND11_MODULE(_winnex_xfactor_c, m) {
    m.doc() = "winnex-xfactor-c — native Ψ attention (latent coupling + phase)";

    // ── Weight derivation (adaptivity) ────────────────────────────────────
    m.def("derive_weights",
          [](py::array_t<float, py::array::c_style | py::array::forcecast> W_q,
             py::array_t<float, py::array::c_style | py::array::forcecast> W_k,
             py::array_t<float, py::array::c_style | py::array::forcecast> W_v,
             int d, int d_q, int d_kv, unsigned seed) {
              auto aq = W_q.request(), ak = W_k.request(), av = W_v.request();
              DerivedWeights dw = derive_weights(
                  (const float*)aq.ptr, (const float*)ak.ptr, (const float*)av.ptr,
                  d, d_q, d_kv, seed);
              auto arr = [](const std::vector<float>& v, int rows, int cols) {
                  py::array_t<float> out(std::vector<py::ssize_t>{(py::ssize_t)rows,
                                                                   (py::ssize_t)cols});
                  std::memcpy(out.mutable_data(), v.data(),
                              (size_t)rows * cols * sizeof(float));
                  return out;
              };
              return py::make_tuple(
                  arr(dw.W_z, 4 * d, d),
                  arr(dw.W_q_prime, d_q, 4 * d),
                  arr(dw.W_r_prime, d_kv, 4 * d),
                  arr(dw.W_h_prime, d_kv, 4 * d));
          },
          py::arg("W_q"), py::arg("W_k"), py::arg("W_v"),
          py::arg("d"), py::arg("d_q"), py::arg("d_kv"),
          py::arg("seed") = 42);

    m.def("preservation_error",
          [](py::array_t<float, py::array::c_style | py::array::forcecast> W_q,
             py::array_t<float, py::array::c_style | py::array::forcecast> W_z,
             py::array_t<float, py::array::c_style | py::array::forcecast> W_qp,
             int max_rows) {
              auto aq = W_q.request(), az = W_z.request(), ap = W_qp.request();
              int d_q = (int)aq.shape[0], d = (int)aq.shape[1];
              DerivedWeights dw;
              dw.d = d; dw.d_q = d_q;
              dw.W_z.assign((const float*)az.ptr, (const float*)az.ptr + az.size);
              dw.W_q_prime.assign((const float*)ap.ptr, (const float*)ap.ptr + ap.size);
              return preservation_error(dw, (const float*)aq.ptr, max_rows);
          },
          py::arg("W_q"), py::arg("W_z"), py::arg("W_qp"),
          py::arg("max_rows") = 0);

    // ── Phase activation ──────────────────────────────────────────────────
    m.def("phase_factors",
          [](py::array_t<float, py::array::c_style | py::array::forcecast> v,
             int n_heads, int head_dim,
             py::object W_phi, int phi_stride) {
              auto info = v.request();
              int H = (int)info.shape[0];
              const float* phi_ptr = nullptr;
              py::array_t<float> Wp;
              if (!W_phi.is_none()) {
                  Wp = W_phi.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
                  phi_ptr = (const float*)Wp.request().ptr;
              }
              py::array_t<float> re(H), im(H);
              phase_factors((const float*)info.ptr, n_heads, head_dim,
                            phi_ptr, phi_stride, re.mutable_data(), im.mutable_data());
              return py::make_tuple(re, im);
          },
          py::arg("v"), py::arg("n_heads"), py::arg("head_dim"),
          py::arg("W_phi") = py::none(), py::arg("phi_stride") = 0);

    // ── Latent coupling forward ───────────────────────────────────────────
    m.def("latent_forward",
          [](py::array_t<float, py::array::c_style | py::array::forcecast> x,
             py::array_t<float, py::array::c_style | py::array::forcecast> W_z,
             py::array_t<float, py::array::c_style | py::array::forcecast> W_qp,
             py::array_t<float, py::array::c_style | py::array::forcecast> W_rp,
             py::array_t<float, py::array::c_style | py::array::forcecast> W_hp,
             int n_heads, int n_kv_heads, int head_dim,
             double phase_alpha, py::object W_phi, int phi_stride) {
              auto ax = x.request(), az = W_z.request();
              auto aq = W_qp.request(), ar = W_rp.request(), ah = W_hp.request();
              int d = (int)ax.shape[0];
              int d_q = n_heads * head_dim, d_kv = n_kv_heads * head_dim;
              // Fail fast on shape mismatch (protects against buffer overflow).
              auto require = [&](const py::buffer_info& a, size_t want, const char* nm) {
                  if ((size_t)a.size != want)
                      throw std::runtime_error(std::string("latent_forward: ") + nm +
                                               " has " + std::to_string(a.size) +
                                               " elements, expected " + std::to_string(want));
              };
              require(az, (size_t)4 * d * d, "W_z");
              require(aq, (size_t)d_q * 4 * d, "W_q'");
              require(ar, (size_t)d_kv * 4 * d, "W_r'");
              require(ah, (size_t)d_kv * 4 * d, "W_h'");
              const float* phi_ptr = nullptr;
              py::array_t<float> Wp;
              if (!W_phi.is_none()) {
                  Wp = W_phi.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
                  phi_ptr = (const float*)Wp.request().ptr;
              }
              py::array_t<float> Z(4 * d), Q(2 * d_q), R(2 * d_kv), H(d_kv);
              latent_forward((const float*)ax.ptr, d, (const float*)az.ptr,
                             (const float*)aq.ptr, (const float*)ar.ptr,
                             (const float*)ah.ptr,
                             n_heads, n_kv_heads, head_dim,
                             phase_alpha, phi_ptr, phi_stride,
                             Z.mutable_data(), Q.mutable_data(),
                             R.mutable_data(), H.mutable_data());
              return py::make_tuple(Z, Q, R, H);
          },
          py::arg("x"), py::arg("W_z"), py::arg("W_qp"), py::arg("W_rp"),
          py::arg("W_hp"), py::arg("n_heads"), py::arg("n_kv_heads"),
          py::arg("head_dim"), py::arg("phase_alpha") = 0.0,
          py::arg("W_phi") = py::none(), py::arg("phi_stride") = 0);

    // ── Phase-modulated score + L2 normalization ──────────────────────────
    m.def("score_phase",
          [](py::array_t<float, py::array::c_style | py::array::forcecast> Q,
             int n_heads, int n_kv_heads, int head_dim, int q_head,
             py::array_t<float, py::array::c_style | py::array::forcecast> R_cache) {
              return score_phase((const float*)Q.request().ptr, n_heads, n_kv_heads,
                                 head_dim, q_head, (const float*)R_cache.request().ptr);
          },
          py::arg("Q"), py::arg("n_heads"), py::arg("n_kv_heads"),
          py::arg("head_dim"), py::arg("q_head"), py::arg("R_cache"));

    // ── HilbertAttention (OpenCL, required — no CPU fallback) ─────────────
    py::class_<HilbertAttention>(m, "HilbertAttention")
        .def(py::init<int, int, int, int, int, int>(),
             py::arg("d_q"), py::arg("d_kv"), py::arg("n_heads"),
             py::arg("n_kv_heads"), py::arg("head_dim"), py::arg("d"))
        .def("set_output_projection",
             [](HilbertAttention& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> o) {
                 self.set_output_projection((const float*)o.request().ptr);
             },
             py::arg("o_proj"))
        .def("kv_append",
             [](HilbertAttention& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> R,
                py::array_t<float, py::array::c_style | py::array::forcecast> H) {
                 self.kv_append((const float*)R.request().ptr,
                                (const float*)H.request().ptr);
             },
             py::arg("R"), py::arg("H"))
        .def("set_latent_weights",
             [](HilbertAttention& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> W_z,
                py::array_t<float, py::array::c_style | py::array::forcecast> W_qp,
                py::array_t<float, py::array::c_style | py::array::forcecast> W_rp,
                py::array_t<float, py::array::c_style | py::array::forcecast> W_hp) {
                 self.set_latent_weights((const float*)W_z.request().ptr,
                                         (const float*)W_qp.request().ptr,
                                         (const float*)W_rp.request().ptr,
                                         (const float*)W_hp.request().ptr);
             },
             py::arg("W_z"), py::arg("W_qp"), py::arg("W_rp"), py::arg("W_hp"))
        .def("set_phase_matrix",
             [](HilbertAttention& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> W_phi) {
                 self.set_phase_matrix((const float*)W_phi.request().ptr);
             },
             py::arg("W_phi"))
        .def("set_spectral_alpha", &HilbertAttention::set_spectral_alpha,
             py::arg("alpha"))
        .def("latent_forward_gpu",
             [](HilbertAttention& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> x,
                double phase_alpha) {
                 py::array_t<float> Q(self.d_q() * 2), R(self.d_kv() * 2);
                 self.latent_forward_gpu((const float*)x.request().ptr,
                                         Q.mutable_data(), R.mutable_data(),
                                         phase_alpha);
                 return py::make_tuple(Q, R);
             },
             py::arg("x"), py::arg("phase_alpha") = 0.0)
        .def("kv_append_gpu", &HilbertAttention::kv_append_gpu)
        .def("forward_gpu",
             [](HilbertAttention& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> x,
                double phase_alpha) {
                 py::array_t<float> y(self.d());
                 self.forward_gpu((const float*)x.request().ptr,
                                  y.mutable_data(), phase_alpha);
                 return y;
             },
             py::arg("x"), py::arg("phase_alpha") = 0.0)
        .def("attention",
             [](HilbertAttention& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> Q) {
                 py::array_t<float> out(self.d_q()), y(self.d());
                 self.attention((const float*)Q.request().ptr,
                                out.mutable_data(), y.mutable_data());
                 return py::make_tuple(out, y);
             },
             py::arg("Q"))
        .def("ctx_len", &HilbertAttention::ctx_len)
        .def("reset", &HilbertAttention::reset)
        .def_property_readonly("device_name",
             [](const HilbertAttention& self) { return std::string(self.device_name()); });

    // ── PsiForwardEngine (full-model Ψ forward) ────────────────────────────
    py::class_<PsiForwardEngine>(m, "PsiForwardEngine")
        .def(py::init<std::string, std::string>(),
             py::arg("model_dir"), py::arg("cache_path") = "")
        .def("forward",
             [](PsiForwardEngine& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> hidden,
                int seq_len, bool all_positions) {
                 auto v = std::vector<float>((const float*)hidden.request().ptr,
                                             (const float*)hidden.request().ptr +
                                             hidden.request().size);
                 auto logits = self.forward(v, seq_len, all_positions);
                 return py::array_t<float>((py::ssize_t)logits.size(), logits.data());
             },
             py::arg("hidden"), py::arg("seq_len"), py::arg("all_positions") = false)
        .def("set_spectral_alpha", &PsiForwardEngine::set_spectral_alpha, py::arg("alpha"))
        .def("set_phase_alpha", &PsiForwardEngine::set_phase_alpha, py::arg("alpha"))
        .def("set_positional", &PsiForwardEngine::set_positional, py::arg("on"))
        .def_property_readonly("num_layers", &PsiForwardEngine::num_layers)
        .def_property_readonly("hidden_size", &PsiForwardEngine::hidden_size)
        .def_property_readonly("vocab_size", &PsiForwardEngine::vocab_size);
}
