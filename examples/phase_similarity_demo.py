"""
Phase Similarity Attention Demo

This script demonstrates the enhanced workspace with phase similarity attention bias.
It compares the standard workspace with the enhanced workspace on a simple task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from core.thalamus import SyntheticThalamus
from core.enhanced_workspace import EnhancedWorkspace
# Define a simple workspace class (instead of importing from train.py)
class SimpleWorkspace(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [B, k, input_dim]
        x_mean = x.mean(dim=1)   # Aggregate over tokens
        hidden = F.relu(self.fc1(x_mean))
        output = self.fc2(hidden)
        return output, x_mean

def create_synthetic_data(batch_size=4, num_tokens=100, d_model=128, phase_dim=16):
    """Create synthetic data for demonstration."""
    # Create content tokens with more variation for better visualization
    tokens = torch.randn(batch_size, num_tokens, d_model)
    
    # Create synthetic task IDs
    task_ids = torch.randint(0, 10, (batch_size,))
    
    # Create synthetic targets
    targets = torch.randint(0, 10, (batch_size,))
    
    return tokens, task_ids, targets

def visualize_attention_patterns(model, tokens, task_ids, enhanced=True):
    """Visualize attention patterns in the workspace."""
    with torch.no_grad():
        # Encode tokens
        encoded = model.encoder(tokens)
        
        # Apply thalamus
        gated = model.thalamus(encoded, task_ids)
        
        # Calculate D (content dimension)
        D = tokens.size(-1)
        
        # Forward through workspace
        if enhanced:
            # Enhanced workspace stores attention weights
            logits, _ = model.workspace(gated)
            attention_weights = model.workspace.attention_weights[0]  # Get first layer's weights
            
            # Average across heads and batch
            attn_avg = attention_weights.mean(dim=1).mean(dim=0).cpu().numpy()
            
            # Extract phase information for visualization
            phase = gated[..., D:].cpu().numpy()[0]  # Take first batch item
        else:
            # Standard workspace doesn't have attention weights
            logits, _ = model.workspace(gated)
            attn_avg = None
            phase = gated[..., D:].cpu().numpy()[0]
        
        # Print phase stats for debugging
        print(f"Phase shape: {phase.shape}")
        print(f"Phase min: {phase.min():.4f}, max: {phase.max():.4f}, mean: {phase.mean():.4f}")
        
        # Ensure some variation in the data for visualization
        if abs(phase.max() - phase.min()) < 1e-6:
            print("Warning: Phase data has almost no variation")
            # Add small random variation for demonstration
            phase = phase + np.random.uniform(-0.1, 0.1, phase.shape)
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    if attn_avg is not None:
        # Plot attention heatmap
        plt.subplot(1, 2, 1)
        plt.imshow(attn_avg, cmap='viridis')
        plt.colorbar(label='Attention Weight')
        plt.title('Attention Pattern (Enhanced Workspace)')
        plt.xlabel('Target Token')
        plt.ylabel('Source Token')
    
    # Alternative visualization for phase vectors
    plt.subplot(1, 2, 2)
    
    # Plot each phase vector as a line
    for i in range(min(phase.shape[0], 8)):  # Plot up to 8 tokens
        plt.plot(range(phase.shape[1]), phase[i], label=f'Token {i}', marker='o')
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel('Phase Dimension')
    plt.ylabel('Phase Value')
    plt.title('Phase Vectors')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"attention_{'enhanced' if enhanced else 'standard'}.png")
    plt.close()
    
    print(f"Visualization saved to 'attention_{'enhanced' if enhanced else 'standard'}.png'")

def main():
    # Parameters
    d_model = 128
    num_tokens = 100
    batch_size = 1
    output_dim = 10
    
    # Create synthetic data
    tokens, task_ids, targets = create_synthetic_data(batch_size, num_tokens, d_model)
    
    # Initialize thalamus with stronger phase modulation
    # This makes the phase variations more visible
    thalamus_params = {
        'd_model': d_model,
        'n_heads': 4,
        'k': 8,
        'phase_dim': 16,
        'task_dim': 64,
        'num_tasks': 10
    }
    
    # Create models with standard and enhanced workspace
    standard_model = torch.nn.Module()
    standard_model.encoder = nn.Linear(d_model, d_model)
    standard_model.thalamus = SyntheticThalamus(**thalamus_params)
    standard_model.workspace = SimpleWorkspace(
        input_dim=d_model + 16, hidden_dim=256, output_dim=output_dim
    )
    
    enhanced_model = torch.nn.Module()
    enhanced_model.encoder = nn.Linear(d_model, d_model)
    enhanced_model.thalamus = SyntheticThalamus(**thalamus_params)
    enhanced_model.workspace = EnhancedWorkspace(
        input_dim=d_model, hidden_dim=256, output_dim=output_dim,
        nhead=4, phase_dim=16, num_layers=2
    )
    
    # Set models to evaluation mode
    standard_model.eval()
    enhanced_model.eval()
    
    # Make the phases more visible (increase magnitude of learned frequencies)
    with torch.no_grad():
        for model in [standard_model, enhanced_model]:
            # Initialize phase_freqs with larger values for better visualization
            model.thalamus.phase_freqs.data = torch.linspace(0.5, 2.0, 16)
    
    # Visualize attention patterns
    print("Visualizing standard workspace attention pattern...")
    visualize_attention_patterns(standard_model, tokens, task_ids, enhanced=False)
    
    print("Visualizing enhanced workspace attention pattern...")
    visualize_attention_patterns(enhanced_model, tokens, task_ids, enhanced=True)
    
    print("Demo complete!")

if __name__ == "__main__":
    main()
