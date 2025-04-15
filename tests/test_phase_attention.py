import torch
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to the system path to import from core
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from core.enhanced_workspace import PhaseSimilarityTransformerLayer, EnhancedWorkspace

def test_phase_similarity_layer_shape():
    """Test that the PhaseSimilarityTransformerLayer maintains the expected output shape."""
    d_model = 128
    phase_dim = 16
    nhead = 4
    batch_size = 2
    seq_len = 8
    
    # Create test inputs
    content = torch.randn(batch_size, seq_len, d_model)
    phase = torch.randn(batch_size, seq_len, phase_dim)
    
    # Initialize layer
    layer = PhaseSimilarityTransformerLayer(d_model, nhead, phase_dim)
    
    # Pass through layer
    output, attn_weights = layer(content, phase)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    
    # Check attention weights shape
    assert attn_weights.shape == (batch_size, nhead, seq_len, seq_len), f"Expected shape {(batch_size, nhead, seq_len, seq_len)}, got {attn_weights.shape}"
    
    print("Phase similarity layer output shape test passed!")

def test_phase_similarity_bias():
    """Test that the phase similarity bias affects attention weights as expected."""
    d_model = 128
    phase_dim = 16
    nhead = 4
    batch_size = 1  # Single sample for easier analysis
    seq_len = 4     # Small sequence length for visualization
    
    # Create content tokens
    content = torch.randn(batch_size, seq_len, d_model)
    
    # Create phase tokens with a specific pattern:
    # - Tokens 0 and 1 have similar phase
    # - Tokens 2 and 3 have similar phase
    # - The two groups have dissimilar phase
    phase = torch.zeros(batch_size, seq_len, phase_dim)
    phase[0, 0, :] = torch.tensor([1.0, 0.0, 0.0] + [0.0] * (phase_dim - 3))
    phase[0, 1, :] = torch.tensor([0.9, 0.1, 0.0] + [0.0] * (phase_dim - 3))  # Similar to token 0
    phase[0, 2, :] = torch.tensor([0.0, 0.0, 1.0] + [0.0] * (phase_dim - 3))
    phase[0, 3, :] = torch.tensor([0.1, 0.0, 0.9] + [0.0] * (phase_dim - 3))  # Similar to token 2
    
    # Initialize layer with phase_scale=0 (no phase bias)
    layer_no_bias = PhaseSimilarityTransformerLayer(d_model, nhead, phase_dim)
    with torch.no_grad():
        layer_no_bias.phase_scale.data = torch.tensor(0.0)
    
    # Initialize layer with phase_scale=10 (strong phase bias)
    layer_with_bias = PhaseSimilarityTransformerLayer(d_model, nhead, phase_dim)
    with torch.no_grad():
        layer_with_bias.phase_scale.data = torch.tensor(10.0)
    
    # Forward pass
    _, attn_no_bias = layer_no_bias(content, phase)
    _, attn_with_bias = layer_with_bias(content, phase)
    
    # Average attention weights across heads for visualization
    attn_no_bias_avg = attn_no_bias.mean(dim=1)[0].detach().numpy()
    attn_with_bias_avg = attn_with_bias.mean(dim=1)[0].detach().numpy()
    
    # Compute phase similarity directly
    phase_norm = torch.nn.functional.normalize(phase, p=2, dim=-1)
    phase_sim = torch.bmm(phase_norm, phase_norm.transpose(1, 2))[0].detach().numpy()
    
    # Check that the similarity pattern appears in the attention with bias
    # Token 0 should attend more to token 1, and token 2 should attend more to token 3
    token_pairs = [(0, 1), (2, 3)]
    for i, j in token_pairs:
        assert attn_with_bias_avg[i, j] > attn_no_bias_avg[i, j], f"Token {i} should attend more to token {j} with phase bias"
    
    # Visualize the attention patterns
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(phase_sim, vmin=0, vmax=1, cmap='viridis')
    plt.colorbar(label='Phase Similarity')
    plt.title('Phase Similarity Matrix')
    plt.xlabel('Token Index')
    plt.ylabel('Token Index')
    
    plt.subplot(1, 3, 2)
    plt.imshow(attn_no_bias_avg, vmin=0, vmax=1, cmap='viridis')
    plt.colorbar(label='Attention Weight')
    plt.title('Attention Weights (No Phase Bias)')
    plt.xlabel('Token Index')
    plt.ylabel('Token Index')
    
    plt.subplot(1, 3, 3)
    plt.imshow(attn_with_bias_avg, vmin=0, vmax=1, cmap='viridis')
    plt.colorbar(label='Attention Weight')
    plt.title('Attention Weights (With Phase Bias)')
    plt.xlabel('Token Index')
    plt.ylabel('Token Index')
    
    plt.tight_layout()
    plt.savefig('phase_attention_test.png')
    
    print("Phase similarity bias test passed!")
    print("Visualization saved to 'phase_attention_test.png'")

def test_enhanced_workspace():
    """Test the EnhancedWorkspace end-to-end."""
    input_dim = 128
    hidden_dim = 256
    output_dim = 10
    phase_dim = 16
    batch_size = 2
    k = 8
    
    # Create test input (simulating output from SyntheticThalamus)
    x_with_phase = torch.randn(batch_size, k, input_dim + phase_dim)
    
    # Initialize EnhancedWorkspace
    workspace = EnhancedWorkspace(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        phase_dim=phase_dim
    )
    
    # Forward pass
    output, pooled = workspace(x_with_phase)
    
    # Check output shapes
    assert output.shape == (batch_size, output_dim), f"Expected output shape {(batch_size, output_dim)}, got {output.shape}"
    assert pooled.shape == (batch_size, input_dim), f"Expected pooled shape {(batch_size, input_dim)}, got {pooled.shape}"
    
    # Check that attention weights were stored
    assert len(workspace.attention_weights) > 0, "No attention weights were stored"
    assert workspace.attention_weights[0].shape == (batch_size, 4, k, k), f"Expected attention weight shape {(batch_size, 4, k, k)}, got {workspace.attention_weights[0].shape}"
    
    print("Enhanced workspace test passed!")

if __name__ == '__main__':
    print("Testing PhaseSimilarityTransformerLayer and EnhancedWorkspace...")
    test_phase_similarity_layer_shape()
    test_phase_similarity_bias()
    test_enhanced_workspace()
    print("All tests passed!")
