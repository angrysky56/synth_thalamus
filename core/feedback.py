# core/feedback.py
import torch
import torch.nn as nn

class FeedbackLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.reward_proj = nn.Linear(d_model, 1)  # Outputs a scalar delta.

    def forward(self, wm_state):
        # wm_state: [B, d_model] global workspace state.
        delta_salience = self.reward_proj(wm_state)
        return delta_salience