"""
Phase Generator Parameter Sweep Demo

This script demonstrates how to use the parameter sweep utilities to find
optimal configurations for the enhanced semantic phase generator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import argparse
from datetime import datetime

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from core.parameter_sweep import PhaseGeneratorSweep
from core.phase_generator import EnhancedSemanticPhaseGenerator
from core.thalamus import SyntheticThalamus
from core.visualization import plot_phase_similarity_matrix, plot_intra_inter_similarity


def create_synthetic_data(batch_size=4, num_tokens=32, d_model=128, num_categories=4, seed=42):
    """
    Create synthetic token embeddings with category information for testing.
    
    Args:
        batch_size: Number of batches
        num_tokens: Number of tokens per batch
        d_model: Dimensionality of token embeddings
        num_categories: Number of semantic categories
        seed: Random seed for reproducibility
        
    Returns:
        tokens: Tensor of shape [batch_size, num_tokens, d_model]
        categories: Tensor of shape [batch_size, num_tokens] with category IDs
    """
    torch.manual_seed(seed)
    
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
            tokens[b, start_idx:end_idx] += category_direction.unsqueeze(0) * 1.0
            
            # Assign category IDs
            categories[b, start_idx:end_idx] = c
    
    return tokens, categories


def run_basic_sweep(tokens, categories, output_dir):
    """
    Run a basic parameter sweep and visualize the results.
    
    Args:
        tokens: Tensor of shape [batch_size, num_tokens, d_model]
        categories: Tensor of shape [batch_size, num_tokens] with category IDs
        output_dir: Directory to save output visualizations
    """
    d_model = tokens.shape[-1]
    phase_dim = 16
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize the sweep
    print("Initializing parameter sweep...")
    sweeper = PhaseGeneratorSweep(d_model=d_model, phase_dim=phase_dim)
    
    # Define parameter ranges for the sweep
    hidden_dims_list = [
        [64],               # 1 layer, small
        [128],              # 1 layer, medium
        [128, 64],          # 2 layers, decreasing
        [64, 128],          # 2 layers, increasing
        [256, 128, 64],     # 3 layers, decreasing
        [64, 128, 256]      # 3 layers, increasing
    ]
    
    activations = ['relu', 'gelu', 'leaky_relu', 'silu']
    diversity_values = [0.5, 1.0, 2.0, 5.0]
    layer_norm_values = [True, False]
    
    # Run the sweep
    print("Running parameter sweep...")
    results = sweeper.run_sweep(
        tokens=tokens,
        categories=categories,
        hidden_dims_list=hidden_dims_list,
        activations=activations,
        diversity_values=diversity_values,
        layer_norm_values=layer_norm_values
    )
    
    # Save the results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"sweep_results_{timestamp}.json")
    sweeper.save_results(results_path)
    
    # Rank configurations by contrast (intra_similarity - inter_similarity)
    print("\nTop configurations by category contrast:")
    try:
        ranked_configs = sweeper.rank_configurations(metric_name='contrast', ascending=False)
        
        for i, (config, metric_value) in enumerate(ranked_configs[:5]):
            print(f"{i+1}. Contrast: {metric_value:.4f}")
            print(f"   Hidden dims: {config['hidden_dims']}")
            print(f"   Activation: {config['activation']}")
            print(f"   Diversity: {config['phase_diversity']}")
            print(f"   Layer norm: {config['use_layer_norm']}")
    except ValueError as e:
        print(f"Could not rank by contrast: {e}")
    
    # Create and save visualizations
    
    # 1. Distribution of contrast values
    print("\nCreating visualizations...")
    try:
        fig_contrast = sweeper.plot_metric_distribution(
            metric_name='contrast',
            title='Distribution of Category Contrast Values'
        )
        fig_contrast.savefig(os.path.join(output_dir, f"contrast_distribution_{timestamp}.png"))
        plt.close(fig_contrast)
    except ValueError as e:
        print(f"Could not plot contrast distribution: {e}")
    
    # 2. Impact of diversity parameter
    try:
        fig_diversity = sweeper.plot_parameter_impact(
            param_name='phase_diversity',
            metric_name='contrast'
        )
        fig_diversity.savefig(os.path.join(output_dir, f"diversity_impact_{timestamp}.png"))
        plt.close(fig_diversity)
    except ValueError as e:
        print(f"Could not plot diversity impact: {e}")
    
    # 3. Impact of activation function
    try:
        fig_activation = sweeper.plot_parameter_impact(
            param_name='activation',
            metric_name='contrast'
        )
        fig_activation.savefig(os.path.join(output_dir, f"activation_impact_{timestamp}.png"))
        plt.close(fig_activation)
    except ValueError as e:
        print(f"Could not plot activation impact: {e}")
    
    # 4. Impact of layer normalization
    try:
        fig_layernorm = sweeper.plot_parameter_impact(
            param_name='use_layer_norm',
            metric_name='contrast'
        )
        fig_layernorm.savefig(os.path.join(output_dir, f"layernorm_impact_{timestamp}.png"))
        plt.close(fig_layernorm)
    except ValueError as e:
        print(f"Could not plot layer norm impact: {e}")
    
    print(f"All visualizations saved to {output_dir}")


def evaluate_best_configuration(tokens, categories, output_dir):
    """
    Evaluate the best configuration found in a previous sweep.
    
    Args:
        tokens: Tensor of shape [batch_size, num_tokens, d_model]
        categories: Tensor of shape [batch_size, num_tokens] with category IDs
        output_dir: Directory to save output visualizations
    """
    d_model = tokens.shape[-1]
    phase_dim = 16
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define best configuration (should be determined from sweep)
    best_config = {
        'hidden_dims': [256, 128, 64],
        'activation': 'gelu',
        'phase_diversity': 2.0,
        'use_layer_norm': True
    }
    
    # Create generator with best configuration
    best_generator = EnhancedSemanticPhaseGenerator(
        d_model=d_model,
        phase_dim=phase_dim,
        **best_config
    )
    
    # Generate phase vectors
    with torch.no_grad():
        phases = best_generator(tokens)
    
    # Create visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Phase similarity matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_phase_similarity_matrix(phases[0], ax=ax, title="Best Configuration Phase Similarity")
    fig.savefig(os.path.join(output_dir, f"best_config_similarity_{timestamp}.png"))
    plt.close(fig)
    
    # 2. Intra vs Inter category similarities
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_intra_inter_similarity(
        phases, categories, ax=ax,
        title="Category Similarity Analysis (Best Configuration)"
    )
    fig.savefig(os.path.join(output_dir, f"best_config_categories_{timestamp}.png"))
    plt.close(fig)
    
    # 3. Visualize individual phase vectors
    fig, ax = plt.subplots(figsize=(12, 8))
    for i in range(min(phases.shape[1], 16)):
        category = categories[0, i].item()
        ax.plot(range(phase_dim), phases[0, i].cpu().numpy(), 
               marker='o', alpha=0.7, 
               label=f"Category {category}" if i < 8 else None)
    
    ax.set_title("Phase Vectors (Best Configuration)")
    ax.set_xlabel("Phase Dimension")
    ax.set_ylabel("Phase Value")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    
    fig.savefig(os.path.join(output_dir, f"best_config_vectors_{timestamp}.png"))
    plt.close(fig)
    
    print(f"Best configuration visualizations saved to {output_dir}")


def load_and_visualize_results(results_path, output_dir):
    """
    Load sweep results and create additional visualizations.
    
    Args:
        results_path: Path to the saved results JSON file
        output_dir: Directory to save output visualizations
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load results
    sweeper = PhaseGeneratorSweep(d_model=128, phase_dim=16)  # Default values, will be overridden by loaded results
    results = sweeper.load_results(results_path)
    
    # Create combined visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a grid of parameter impact plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Impact of diversity parameter
    try:
        ax = axes[0, 0]
        param_values = []
        metric_means = []
        metric_stds = []
        
        # Extract metric values grouped by diversity parameter
        diversity_values = sorted(set(config['phase_diversity'] for config in results['configurations']))
        for diversity in diversity_values:
            metrics = []
            for i, config in enumerate(results['configurations']):
                if config['phase_diversity'] == diversity:
                    metrics.append(results['metrics'][i]['contrast'])
            
            param_values.append(diversity)
            metric_means.append(np.mean(metrics))
            metric_stds.append(np.std(metrics))
        
        # Plot
        ax.errorbar(param_values, metric_means, yerr=metric_stds, marker='o', linestyle='-')
        ax.set_xlabel('Phase Diversity')
        ax.set_ylabel('Category Contrast')
        ax.set_title('Impact of Phase Diversity on Category Contrast')
        ax.grid(True, linestyle='--', alpha=0.7)
    except Exception as e:
        print(f"Could not create diversity impact plot: {e}")
    
    # Impact of layer norm
    try:
        ax = axes[0, 1]
        
        # Extract metric values grouped by layer norm
        layernorm_values = [True, False]
        layernorm_labels = ['With LayerNorm', 'Without LayerNorm']
        metric_means = []
        metric_stds = []
        
        for use_layernorm in layernorm_values:
            metrics = []
            for i, config in enumerate(results['configurations']):
                if config['use_layer_norm'] == use_layernorm:
                    metrics.append(results['metrics'][i]['contrast'])
            
            metric_means.append(np.mean(metrics))
            metric_stds.append(np.std(metrics))
        
        # Plot
        x = np.arange(len(layernorm_values))
        ax.bar(x, metric_means, yerr=metric_stds, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(layernorm_labels)
        ax.set_ylabel('Category Contrast')
        ax.set_title('Impact of Layer Normalization on Category Contrast')
        ax.grid(True, linestyle='--', alpha=0.7)
    except Exception as e:
        print(f"Could not create layer norm impact plot: {e}")
    
    # Impact of activation function
    try:
        ax = axes[1, 0]
        
        # Extract metric values grouped by activation
        activation_values = sorted(set(config['activation'] for config in results['configurations']))
        metric_means = []
        metric_stds = []
        
        for activation in activation_values:
            metrics = []
            for i, config in enumerate(results['configurations']):
                if config['activation'] == activation:
                    metrics.append(results['metrics'][i]['contrast'])
            
            metric_means.append(np.mean(metrics))
            metric_stds.append(np.std(metrics))
        
        # Plot
        x = np.arange(len(activation_values))
        ax.bar(x, metric_means, yerr=metric_stds, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(activation_values)
        ax.set_ylabel('Category Contrast')
        ax.set_title('Impact of Activation Function on Category Contrast')
        ax.grid(True, linestyle='--', alpha=0.7)
    except Exception as e:
        print(f"Could not create activation impact plot: {e}")
    
    # Impact of network depth
    try:
        ax = axes[1, 1]
        
        # Extract metric values grouped by network depth
        depths = []
        for config in results['configurations']:
            depth = len(config['hidden_dims'])
            if depth not in depths:
                depths.append(depth)
        
        depths.sort()
        metric_means = []
        metric_stds = []
        
        for depth in depths:
            metrics = []
            for i, config in enumerate(results['configurations']):
                if len(config['hidden_dims']) == depth:
                    metrics.append(results['metrics'][i]['contrast'])
            
            metric_means.append(np.mean(metrics))
            metric_stds.append(np.std(metrics))
        
        # Plot
        ax.errorbar(depths, metric_means, yerr=metric_stds, marker='o', linestyle='-')
        ax.set_xlabel('Network Depth (Number of Hidden Layers)')
        ax.set_ylabel('Category Contrast')
        ax.set_title('Impact of Network Depth on Category Contrast')
        ax.grid(True, linestyle='--', alpha=0.7)
    except Exception as e:
        print(f"Could not create network depth impact plot: {e}")
    
    # Save combined plot
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"parameter_impacts_combined_{timestamp}.png"))
    plt.close(fig)
    
    print(f"Parameter impact visualizations saved to {output_dir}")


def create_thalamus_with_best_config(d_model=128, n_heads=4, k=16, phase_dim=16,
                                    task_dim=64, num_tasks=10):
    """
    Create a SyntheticThalamus with the best configuration for the phase generator.
    
    Args:
        d_model: Dimension of input feature vectors
        n_heads: Number of attention heads
        k: Top-K tokens to gate
        phase_dim: Dimensionality of the phase tag
        task_dim: Dimensionality of task conditioning
        num_tasks: Number of distinct tasks
        
    Returns:
        thalamus: SyntheticThalamus with optimized phase generator
    """
    # Best configuration (should be determined from sweep)
    best_config = {
        'hidden_dims': [256, 128, 64],
        'activation': 'gelu',
        'phase_diversity': 2.0,
        'use_layer_norm': True
    }
    
    # Create thalamus with best configuration
    thalamus = SyntheticThalamus(
        d_model=d_model,
        n_heads=n_heads,
        k=k,
        phase_dim=phase_dim,
        task_dim=task_dim,
        num_tasks=num_tasks,
        **best_config
    )
    
    return thalamus


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Phase Generator Parameter Sweep Demo')
    parser.add_argument('--mode', type=str, default='sweep',
                      choices=['sweep', 'visualize', 'evaluate'],
                      help='Mode to run: sweep, visualize, or evaluate')
    parser.add_argument('--results', type=str, default=None,
                      help='Path to results JSON file (for visualize mode)')
    parser.add_argument('--output_dir', type=str, default='sweep_results',
                      help='Directory to save output visualizations')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create synthetic data
    print("Creating synthetic token data...")
    tokens, categories = create_synthetic_data(
        batch_size=2,
        num_tokens=32,
        d_model=128,
        num_categories=4
    )
    
    if args.mode == 'sweep':
        print("\nRunning parameter sweep...")
        run_basic_sweep(tokens, categories, args.output_dir)
    
    elif args.mode == 'visualize':
        if args.results is None:
            print("Error: --results argument is required for visualize mode")
            return
        
        print(f"\nLoading and visualizing results from {args.results}...")
        load_and_visualize_results(args.results, args.output_dir)
    
    elif args.mode == 'evaluate':
        print("\nEvaluating best configuration...")
        evaluate_best_configuration(tokens, categories, args.output_dir)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
