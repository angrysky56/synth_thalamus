"""
Enhanced Phase Generator Demo

This script demonstrates the enhanced semantic phase generator with
various configurations and visualizes the results.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from datetime import datetime

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from core.phase_generator import EnhancedSemanticPhaseGenerator, PhaseAnalyzer
from core.visualization import (
    plot_phase_similarity_matrix, 
    visualize_phase_diversity,
    plot_intra_inter_similarity
)

def create_synthetic_tokens(batch_size=1, num_tokens=32, d_model=128, num_categories=4):
    """
    Create synthetic token embeddings with category information for testing.
    
    Args:
        batch_size: Number of batches
        num_tokens: Number of tokens per batch
        d_model: Dimensionality of token embeddings
        num_categories: Number of semantic categories
        
    Returns:
        tokens: Tensor of shape [batch_size, num_tokens, d_model]
        categories: Tensor of shape [batch_size, num_tokens] with category IDs
    """
    tokens = torch.randn(batch_size, num_tokens, d_model)
    
    # Create category IDs (evenly distributed)
    categories = torch.zeros(batch_size, num_tokens, dtype=torch.long)
    tokens_per_category = num_tokens // num_categories
    
    for b in range(batch_size):
        for c in range(num_categories):
            start_idx = c * tokens_per_category
            end_idx = (c + 1) * tokens_per_category if c < num_categories - 1 else num_tokens
            
            # Add category bias to embeddings
            category_direction = torch.randn(d_model)
            tokens[b, start_idx:end_idx] += category_direction.unsqueeze(0) * 0.5
            
            # Assign category IDs
            categories[b, start_idx:end_idx] = c
    
    return tokens, categories

def compare_activation_functions(tokens, categories):
    """
    Compare different activation functions for the phase generator.
    
    Args:
        tokens: Tensor of shape [batch_size, num_tokens, d_model]
        categories: Tensor of shape [batch_size, num_tokens] with category IDs
    """
    d_model = tokens.shape[-1]
    phase_dim = 16
    hidden_dims = [128, 64]
    phase_diversity = 2.0
    
    # Create generators with different activation functions
    activation_functions = ['relu', 'gelu', 'leaky_relu', 'silu', 'prelu']
    generators = []
    
    for activation in activation_functions:
        generator = EnhancedSemanticPhaseGenerator(
            d_model=d_model,
            phase_dim=phase_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            phase_diversity=phase_diversity
        )
        generators.append(generator)
    
    # Create figure
    fig, axes = plt.subplots(len(activation_functions), 2, figsize=(12, 4 * len(activation_functions)))
    
    # Generate and plot phases for each activation function
    for i, (generator, activation) in enumerate(zip(generators, activation_functions)):
        with torch.no_grad():
            phase = generator(tokens)
            
            # Plot phase vectors
            for j in range(min(8, tokens.shape[1])):
                category = categories[0, j].item()
                axes[i, 0].plot(range(phase_dim), phase[0, j].numpy(), 
                               marker='o', alpha=0.7, 
                               label=f"Cat {category}" if j < 4 else None)
                
            axes[i, 0].set_title(f"{activation.upper()} Activation")
            axes[i, 0].set_xlabel("Phase Dimension")
            axes[i, 0].set_ylabel("Phase Value")
            axes[i, 0].grid(True, linestyle='--', alpha=0.5)
            axes[i, 0].legend()
            
            # Plot phase similarity matrix
            plot_phase_similarity_matrix(phase[0], ax=axes[i, 1], 
                                       title=f"{activation.upper()} Similarity")
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"activation_comparison_{timestamp}.png")
    plt.close()
    
    print(f"Activation function comparison saved to activation_comparison_{timestamp}.png")

def compare_diversity_parameters(tokens, categories):
    """
    Compare different diversity parameter values.
    
    Args:
        tokens: Tensor of shape [batch_size, num_tokens, d_model]
        categories: Tensor of shape [batch_size, num_tokens] with category IDs
    """
    d_model = tokens.shape[-1]
    phase_dim = 16
    hidden_dims = [128, 64]
    activation = 'gelu'
    
    # Create generators with different diversity parameters
    diversity_values = [0.5, 1.0, 2.0, 3.0, 5.0]
    generators = []
    
    for diversity in diversity_values:
        generator = EnhancedSemanticPhaseGenerator(
            d_model=d_model,
            phase_dim=phase_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            phase_diversity=diversity
        )
        generators.append(generator)
    
    # Create visualization
    fig = visualize_phase_diversity(generators, tokens, diversity_values,
                                  title="Impact of Phase Diversity Parameter")
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"diversity_comparison_{timestamp}.png")
    plt.close()
    
    print(f"Diversity parameter comparison saved to diversity_comparison_{timestamp}.png")
    
    # Compute and compare intra-category vs inter-category similarities
    fig, axes = plt.subplots(1, len(diversity_values), figsize=(15, 5))
    
    metrics_list = []
    for i, (generator, diversity) in enumerate(zip(generators, diversity_values)):
        with torch.no_grad():
            phase = generator(tokens)
            _, metrics = plot_intra_inter_similarity(
                phase, categories, ax=axes[i], 
                title=f"Diversity = {diversity:.1f}"
            )
            metrics_list.append(metrics)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"category_similarity_{timestamp}.png")
    plt.close()
    
    print(f"Category similarity analysis saved to category_similarity_{timestamp}.png")
    
    # Print metrics summary
    print("\nMetrics Summary:")
    print("----------------")
    print("Diversity | Intra-Sim | Inter-Sim | Contrast")
    print("------------------------------------------")
    for diversity, metrics in zip(diversity_values, metrics_list):
        print(f"{diversity:8.1f} | {metrics['intra_similarity']:8.3f} | {metrics['inter_similarity']:8.3f} | {metrics['contrast']:8.3f}")

def compare_network_architectures(tokens, categories):
    """
    Compare different network architectures for the phase generator.
    
    Args:
        tokens: Tensor of shape [batch_size, num_tokens, d_model]
        categories: Tensor of shape [batch_size, num_tokens] with category IDs
    """
    d_model = tokens.shape[-1]
    phase_dim = 16
    activation = 'gelu'
    phase_diversity = 2.0
    
    # Define different architectures
    architectures = [
        {'name': 'Simple', 'hidden_dims': [64]},
        {'name': 'Medium', 'hidden_dims': [128, 64]},
        {'name': 'Deep', 'hidden_dims': [256, 128, 64]},
        {'name': 'Wide', 'hidden_dims': [512, 256]},
        {'name': 'Very Deep', 'hidden_dims': [256, 128, 64, 32]}
    ]
    
    generators = []
    for arch in architectures:
        generator = EnhancedSemanticPhaseGenerator(
            d_model=d_model,
            phase_dim=phase_dim,
            hidden_dims=arch['hidden_dims'],
            activation=activation,
            phase_diversity=phase_diversity
        )
        generators.append(generator)
    
    # Create figure for phase vector visualization
    fig, axes = plt.subplots(len(architectures), 2, figsize=(12, 4 * len(architectures)))
    
    # Generate and visualize phases for each architecture
    for i, (generator, arch) in enumerate(zip(generators, architectures)):
        with torch.no_grad():
            phase = generator(tokens)
            
            # Plot phase vectors
            for j in range(min(8, tokens.shape[1])):
                category = categories[0, j].item()
                axes[i, 0].plot(range(phase_dim), phase[0, j].numpy(), 
                              marker='o', alpha=0.7, 
                              label=f"Cat {category}" if j < 4 else None)
                
            axes[i, 0].set_title(f"{arch['name']} Architecture: {arch['hidden_dims']}")
            axes[i, 0].set_xlabel("Phase Dimension")
            axes[i, 0].set_ylabel("Phase Value")
            axes[i, 0].grid(True, linestyle='--', alpha=0.5)
            axes[i, 0].legend()
            
            # Plot phase similarity matrix
            plot_phase_similarity_matrix(phase[0], ax=axes[i, 1], 
                                      title=f"{arch['name']} Similarity")
            
            # Calculate similarity metrics
            analyzer = PhaseAnalyzer()
            metrics = analyzer.compute_category_similarities(phase, categories)
            
            # Add metrics to plot
            info_text = (f"Intra-Sim: {metrics['intra_similarity'][0]:.3f}\n"
                       f"Inter-Sim: {metrics['inter_similarity'][0]:.3f}\n"
                       f"Contrast: {metrics['contrast'][0]:.3f}")
            
            axes[i, 1].text(0.05, 0.95, info_text, transform=axes[i, 1].transAxes,
                          verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"architecture_comparison_{timestamp}.png")
    plt.close()
    
    print(f"Architecture comparison saved to architecture_comparison_{timestamp}.png")

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create synthetic data
    print("Creating synthetic token data...")
    tokens, categories = create_synthetic_tokens(
        batch_size=1,
        num_tokens=32,
        d_model=128,
        num_categories=4
    )
    
    print(f"Generated tokens shape: {tokens.shape}")
    print(f"Categories shape: {categories.shape}")
    print(f"Category distribution: {torch.bincount(categories[0])}")
    
    # Run comparisons
    print("\nComparing activation functions...")
    compare_activation_functions(tokens, categories)
    
    print("\nComparing diversity parameters...")
    compare_diversity_parameters(tokens, categories)
    
    print("\nComparing network architectures...")
    compare_network_architectures(tokens, categories)
    
    print("\nAll comparisons completed!")

if __name__ == "__main__":
    main()
