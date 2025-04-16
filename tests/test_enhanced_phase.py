"""
Test Enhanced Semantic Phase Generator

This script tests the functionality of the enhanced semantic phase generator,
including its architecture, activation functions, and diversity parameter.
"""

import torch
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F

# Add the parent directory to the system path to import from core
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from core.phase_generator import EnhancedSemanticPhaseGenerator, contrastive_loss, PhaseAnalyzer

def test_enhanced_generator_shape():
    """Test that the EnhancedSemanticPhaseGenerator maintains the expected output shape."""
    d_model = 128
    phase_dim = 16
    hidden_dims = [128, 64]
    batch_size = 2
    seq_len = 8
    
    # Create test inputs
    inputs = torch.randn(batch_size, seq_len, d_model)
    
    # Initialize generator
    generator = EnhancedSemanticPhaseGenerator(
        d_model=d_model,
        phase_dim=phase_dim,
        hidden_dims=hidden_dims,
        activation='gelu',
        phase_diversity=2.0
    )
    
    # Forward pass
    output = generator(inputs)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, phase_dim), f"Expected shape {(batch_size, seq_len, phase_dim)}, got {output.shape}"
    
    # Check output range
    assert torch.all(output >= -1.0) and torch.all(output <= 1.0), "Output values should be in range [-1, 1]"
    
    print("Enhanced generator output shape test passed!")

def test_contrastive_loss():
    """Test that the contrastive loss correctly encourages similar phases for similar categories."""
    batch_size = 2
    seq_len = 8
    phase_dim = 16
    
    # Create synthetic phase vectors
    phase_vectors = torch.randn(batch_size, seq_len, phase_dim)
    phase_vectors = F.normalize(phase_vectors, p=2, dim=2)  # Normalize for testing
    
    # Create synthetic category IDs (first 4 tokens in category 0, rest in category 1)
    categories = torch.zeros(batch_size, seq_len, dtype=torch.long)
    categories[:, 4:] = 1
    
    # Calculate contrastive loss
    loss = contrastive_loss(phase_vectors, categories, temperature=0.5)
    
    # Check that loss is a scalar
    assert loss.dim() == 0, f"Expected scalar loss, got shape {loss.shape}"
    
    # Check that loss is positive
    assert loss > 0, f"Expected positive loss, got {loss.item()}"
    
    print("Contrastive loss test passed!")

def test_activation_functions():
    """Test different activation functions for the phase generator."""
    d_model = 128
    phase_dim = 16
    hidden_dims = [128, 64]
    batch_size = 1
    seq_len = 8
    
    # Create test inputs
    inputs = torch.randn(batch_size, seq_len, d_model)
    
    # Test each activation function
    activations = ['relu', 'gelu', 'leaky_relu', 'silu', 'prelu']
    
    for activation in activations:
        # Initialize generator with this activation
        generator = EnhancedSemanticPhaseGenerator(
            d_model=d_model,
            phase_dim=phase_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            phase_diversity=2.0
        )
        
        # Forward pass
        output = generator(inputs)
        
        # Calculate statistics
        min_val = output.min().item()
        max_val = output.max().item()
        mean_val = output.mean().item()
        std_val = output.std().item()
        
        # Check output range
        assert min_val >= -1.0 and max_val <= 1.0, f"Output for {activation} exceeds range [-1, 1]"
        
        print(f"Activation {activation}: min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}, std={std_val:.4f}")
    
    print("Activation functions test passed!")

def test_phase_diversity():
    """Test the effect of the phase diversity parameter."""
    d_model = 128
    phase_dim = 16
    hidden_dims = [128, 64]
    batch_size = 1
    seq_len = 8
    
    # Create test inputs
    inputs = torch.randn(batch_size, seq_len, d_model)
    
    # Test different diversity values
    diversity_values = [0.1, 1.0, 3.0, 5.0]
    
    # Collect statistics
    std_values = []
    
    for diversity in diversity_values:
        # Initialize generator with this diversity
        generator = EnhancedSemanticPhaseGenerator(
            d_model=d_model,
            phase_dim=phase_dim,
            hidden_dims=hidden_dims,
            activation='gelu',
            phase_diversity=diversity
        )
        
        # Forward pass
        with torch.no_grad():
            output = generator(inputs)
        
        # Calculate standard deviation (measure of diversity)
        std_val = output.std().item()
        std_values.append(std_val)
        
        print(f"Diversity {diversity}: std={std_val:.4f}")
    
    # Check that higher diversity parameter leads to higher standard deviation
    assert all(std_values[i] <= std_values[i+1] for i in range(len(std_values)-1)), \
        "Higher diversity parameter should lead to higher standard deviation"
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.plot(diversity_values, std_values, marker='o')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel('Phase Diversity Parameter')
    plt.ylabel('Standard Deviation of Phase Values')
    plt.title('Effect of Phase Diversity Parameter')
    plt.savefig('phase_diversity_test.png')
    plt.close()
    
    print("Phase diversity test passed! Visualization saved to 'phase_diversity_test.png'")

def test_phase_analyzer():
    """Test the PhaseAnalyzer for computing similarity metrics."""
    batch_size = 1
    seq_len = 8
    phase_dim = 16
    
    # Create synthetic phase vectors with clear category structure
    phase_vectors = torch.zeros(batch_size, seq_len, phase_dim)
    # First 4 tokens in category 0, with similar phase
    phase_vectors[0, 0:4, 0] = 1.0
    # Last 4 tokens in category 1, with similar phase
    phase_vectors[0, 4:8, 2] = 1.0
    
    # Create category IDs
    categories = torch.zeros(batch_size, seq_len, dtype=torch.long)
    categories[0, 4:] = 1
    
    # Calculate similarity metrics
    analyzer = PhaseAnalyzer()
    metrics = analyzer.compute_similarity_matrix(phase_vectors)
    category_metrics = analyzer.compute_category_similarities(phase_vectors, categories)
    
    # Check that similarity matrix has expected shape
    assert metrics.shape == (batch_size, seq_len, seq_len), f"Expected shape {(batch_size, seq_len, seq_len)}, got {metrics.shape}"
    
    # Check that intra-category similarity is higher than inter-category similarity
    assert category_metrics['intra_similarity'][0] > category_metrics['inter_similarity'][0], \
        "Intra-category similarity should be higher than inter-category similarity"
    
    print(f"Phase analyzer test passed! Intra-sim: {category_metrics['intra_similarity'][0]:.4f}, Inter-sim: {category_metrics['inter_similarity'][0]:.4f}")

if __name__ == '__main__':
    print("Testing EnhancedSemanticPhaseGenerator...")
    test_enhanced_generator_shape()
    test_contrastive_loss()
    test_activation_functions()
    test_phase_diversity()
    test_phase_analyzer()
    print("All tests passed!")
