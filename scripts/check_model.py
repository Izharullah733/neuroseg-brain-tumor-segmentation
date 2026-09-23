"""Synthetic shape/gradient smoke check, not a real-data benchmark."""

import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from neuroseg.model import CompactUNet


if __name__ == '__main__':
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = CompactUNet()
    start = perf_counter()
    for size in [(128, 128), (127, 129)]:
        model.zero_grad(set_to_none=True)
        output = model(torch.randn(1, 4, *size))
        assert output.shape == (1, 4, *size)
        assert torch.isfinite(output).all()
        output.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print(f'PASS: output shapes and finite gradients; parameters={sum(p.numel() for p in model.parameters()):,}')
    print(f'Synthetic checks elapsed: {perf_counter() - start:.2f}s. No real-data quality measured.')
