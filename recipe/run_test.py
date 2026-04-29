import os

os.environ["HELION_AUTOTUNE_EFFORT"] = "none"

import torch
import helion
import helion.language as hl

assert torch.cuda.is_available(), "CUDA device required for helion smoke test"


@helion.kernel(autotune_effort="none")
def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + y[tile]
    return out


x = torch.randn(64, device="cuda", dtype=torch.float32)
y = torch.randn(64, device="cuda", dtype=torch.float32)
result = add_kernel(x, y)
torch.testing.assert_close(result, x + y)
print("helion GPU smoke test passed")
