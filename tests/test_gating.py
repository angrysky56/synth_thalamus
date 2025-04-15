# tests/test_gating.py
import torch
import sys
import os

# Add the parent directory to the system path to import from core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.thalamus import SyntheticThalamus

def test_topk_gating():
    d_model = 128
    batch_size = 2
    num_tokens = 100
    k = 8
    phase_dim = 16
    model = SyntheticThalamus(d_model=d_model, n_heads=4, k=k, phase_dim=phase_dim, task_dim=64, num_tasks=10)
    x = torch.randn(batch_size, num_tokens, d_model)
    task_id = torch.randint(0, 10, (batch_size,))
    gated = model(x, task_id)
    expected_dim = d_model + phase_dim
    assert gated.shape == (batch_size, k, expected_dim), f"Expected shape {(batch_size, k, expected_dim)}, got {gated.shape}"

if __name__ == '__main__':
    test_topk_gating()
    print("test_topk_gating passed.")