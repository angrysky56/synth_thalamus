# tests/test_phase.py
import torch
import sys
import os

# Add the parent directory to the system path to import from core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.thalamus import SyntheticThalamus

def test_phase_generation():
    d_model = 128
    phase_dim = 16
    model = SyntheticThalamus(d_model=d_model, k=8, phase_dim=phase_dim, task_dim=64, num_tasks=10)
    batch_size = 2
    num_tokens = 100
    x = torch.randn(batch_size, num_tokens, d_model)
    task_id = torch.randint(0, 10, (batch_size,))
    gated = model(x, task_id)
    # Extract the phase part (last phase_dim elements)
    phase = gated[..., d_model:]
    assert torch.all(phase <= 1.0) and torch.all(phase >= -1.0), "Phase values should be in [-1,1]"
    
if __name__ == '__main__':
    test_phase_generation()
    print("test_phase_generation passed.")