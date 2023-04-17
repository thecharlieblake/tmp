// All pure C++ headers for the C++ frontend.
#include <torch/all.h>
// Python bindings for the C++ frontend (includes Python.h).
#include <torch/python.h>

using namespace at;
using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class _ScaledGrad : public torch::autograd::Function<_ScaledGrad> {
 public:
  static variable_list forward(
      AutogradContext* ctx,
      Variable X,
      Scalar fwd_scale,
      Scalar bwd_scale) {
    ctx->saved_data["bwd_scale"] = bwd_scale;  // TODO: dtype
    return {X.mul(fwd_scale)};
  }

  static variable_list backward(
      AutogradContext* ctx,
      variable_list grad_Ys) {
    auto bwd_scale = ctx->saved_data["bwd_scale"].toScalar();
    return {grad_Ys[0].mul(bwd_scale), Tensor(), Tensor()};
  }
};

Tensor _scale(const Tensor& input, Scalar fwd_scale, Scalar bwd_scale) {
  return _ScaledGrad::apply(input, fwd_scale, bwd_scale)[0];
}

TORCH_LIBRARY(my_ops, m) {
    m.def("_scale", _scale);
}