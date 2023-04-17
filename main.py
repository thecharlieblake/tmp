from typing import Tuple

import torch
from torch import Tensor

cpp_version = False

if cpp_version:
    from scale_op import _scale
else:

    class _ScaledGrad(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: torch.autograd.function.FunctionCtx,
            X: Tensor,
            fwd_scale: float,
            bwd_scale: float,
        ) -> Tensor:
            ctx.save_for_backward(torch.tensor(bwd_scale, dtype=X.dtype))
            return fwd_scale * X

        @staticmethod
        def backward(
            ctx: torch.autograd.function.FunctionCtx, grad_Y: Tensor
        ) -> Tuple[Tensor, None, None]:
            (bwd_scale,) = ctx.saved_tensors
            return bwd_scale * grad_Y, None, None

    def _scale(t: Tensor, fwd_scale: float = 1.0, bwd_scale: float = 1.0) -> Tensor:
        return _ScaledGrad.apply(t, fwd_scale, bwd_scale)


@torch.jit.script
def f(a):
    return _scale(a @ torch.ones(3, 3), 3.0, 5.0)


x = torch.ones(3).requires_grad_()
y = f(x)

g = torch.ones(3)
y.backward(g)

print(y, x.grad)
