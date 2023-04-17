import glob
import os

import torch
from torch import Tensor

package_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

pattern = (
    os.path.join(package_path, "_scale*.so")
    if os.name != "nt"
    else os.path.join(package_path, "_scale*.pyd")
)
print(pattern)
custom_op_library_path = glob.glob(pattern)[0]

torch.ops.load_library(custom_op_library_path)


def _scale(t: Tensor, fwd: float, bwd: float):
    return torch.ops.my_ops._scale(t, fwd, bwd)
