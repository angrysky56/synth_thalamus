"""
Recurrent Thalamus-Workspace Demo

This script demonstrates the recurrent connection between thalamus and workspace,
creating a feedback loop that allows the workspace to influence thalamus processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import argparse
from tqdm import tqdm
from datetime import datetime

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from core.thalamus import SyntheticThalamus
from core.enhanced_workspace import EnhancedWorkspace
from core.feedback_loops import WorkspaceToThalamusFeedback, RecurrentThalamusWorkspace
from core.visualization import plot_phase_similarity_matrix, plot_intra_inter_similarity
from core.visualization_enhanced import visualize_before_after_comparison


def create_synthetic_data(batch_size=4, num_tokens=100, d_model=128, num_categories=4):
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


def compare_models(tokens, categories, task_ids, output_dir):
    """
    Compare standard vs. recurrent thalamus-workspace setup.
    
    Args:
        tokens: Input tokens [B, N, D]
        categories: Token categories [B, N]
        task_ids: Task identifiers [B]
        output_dir: Directory to save visualizations
    """
    # Parameters
    d_model = tokens.shape[-1]
    phase_dim = 16
    k = 16
    task_dim = 64
    num_tasks = 10
    
    # Create standard thalamus and workspace
    standard_thalamus = SyntheticThalamus(
        d_model=d_model,
        n_heads=4,
        k=k,
        phase_dim=phase_dim,
        task_dim=task_dim,
        num_tasks=num_tasks,
        hidden_dims=[128, 64],
        activation='gelu',
        phase_diversity=2.0,
        use_layer_norm=True
    )
    
    standard_workspace = EnhancedWorkspace(
        input_dim=d_model,
        hidden_dim=256,
        output_dim=10,
        nhead=4,
        phase_dim=phase_dim,
        num_layers=2
    )
    
    # Create feedback module
    feedback = WorkspaceToThalamusFeedback(
        workspace_dim=d_model,
        thalamus_dim=d_model,
        feedback_dim=64,
        num_heads=4,
        use_gating=True
    )
    
    # Create recurrent thalamus-workspace
    recurrent_model = RecurrentThalamusWorkspace(
        thalamus=standard_thalamus,
        workspace=standard_workspace,
        feedback=feedback,
        max_iterations=3,
        adaptive_iterations=False
    )
    
    # Process data through standard pipeline
    print("Processing through standard pipeline...")
    with torch.no_grad():
        # Forward through thalamus
        standard_thalamus_output, standard_indices = standard_thalamus(tokens, task_ids)
        
        # Extract phase vectors for visualization
        standard_phases = standard_thalamus_output[..., d_model:]
        
        # Forward through workspace
        standard_output, standard_pooled = standard_workspace(standard_thalamus_output)
    
    # Process data through recurrent pipeline
    print("Processing through recurrent pipeline...")
    with torch.no_grad():
        # Forward through recurrent model
        recurrent_output, recurrent_pooled, intermediates = recurrent_model(
            tokens, task_ids, return_intermediates=True
        )
        
        # Extract phases from last iteration for comparison
        last_thalamus_output = intermediates['thalamus_outputs'][-1]
        recurrent_phases = last_thalamus_output[..., d_model:]
        
        # Get the token indices for the last iteration if available
        if 'token_indices' in intermediates and intermediates['token_indices']:
            recurrent_indices = intermediates['token_indices'][-1]
        else:
            # Fallback if indices aren't available
            recurrent_indices = standard_indices
    
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Visualize phases from both approaches
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Compare phase similarity matrices
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Standard phases
    plot_phase_similarity_matrix(standard_phases[0], ax=axes[0], 
                              title="Standard Thalamus Phase Similarity")
    
    # Recurrent phases
    plot_phase_similarity_matrix(recurrent_phases[0], ax=axes[1], 
                              title="Recurrent Thalamus Phase Similarity")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"phase_comparison_{timestamp}.png"))
    plt.close()
    
    # Compare intra vs inter category similarities
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Extract the correct categories for the gated tokens using the indices
    standard_gated_categories = torch.zeros_like(standard_indices)
    for b in range(standard_indices.shape[0]):
        standard_gated_categories[b] = torch.gather(categories[b], 0, standard_indices[b])
    
    recurrent_gated_categories = torch.zeros_like(recurrent_indices)
    for b in range(recurrent_indices.shape[0]):
        recurrent_gated_categories[b] = torch.gather(categories[b], 0, recurrent_indices[b])
    
    # Standard
    plot_intra_inter_similarity(
        standard_phases, standard_gated_categories, ax=axes[0],
        title="Standard Thalamus Category Similarity"
    )
    
    # Recurrent
    plot_intra_inter_similarity(
        recurrent_phases, recurrent_gated_categories, ax=axes[1],
        title="Recurrent Thalamus Category Similarity"
    )
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"category_similarity_{timestamp}.png"))
    plt.close()
    
    # Visualize the progression of phases across iterations
    if len(intermediates['thalamus_outputs']) > 1:
        print("Visualizing phase evolution across iterations...")
        
        # Extract phases from each iteration
        iteration_phases = []
        iteration_categories = []
        
        for i, thalamus_output in enumerate(intermediates['thalamus_outputs']):
            phases = thalamus_output[..., d_model:]
            iteration_phases.append(phases)
            
            # Get categories for this iteration
            if 'token_indices' in intermediates and i < len(intermediates['token_indices']):
                indices = intermediates['token_indices'][i]
                iteration_cats = torch.zeros_like(indices)
                for b in range(indices.shape[0]):
                    iteration_cats[b] = torch.gather(categories[b], 0, indices[b])
                iteration_categories.append(iteration_cats)
            else:
                # If indices aren't available, use standard categories as fallback
                iteration_categories.append(standard_gated_categories)
        
        # Create multi-panel visualization
        num_iterations = len(iteration_phases)
        fig, axes = plt.subplots(2, num_iterations, figsize=(6 * num_iterations, 12))
        
        for i, (phases, cats) in enumerate(zip(iteration_phases, iteration_categories)):
            # Phase similarity matrix
            plot_phase_similarity_matrix(phases[0], ax=axes[0, i], 
                                      title=f"Iteration {i+1} Phase Similarity")
            
            # Intra vs inter category similarities
            plot_intra_inter_similarity(
                phases, cats, ax=axes[1, i],
                title=f"Iteration {i+1} Category Similarity"
            )
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"phase_evolution_{timestamp}.png"))
        plt.close()
        
        # Visualize feedback gates if available
        if 'feedback_gates' in intermediates and intermediates['feedback_gates']:
            gates = torch.cat(intermediates['feedback_gates'], dim=1).cpu().numpy()
            
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(intermediates['feedback_gates']) + 1), gates.mean(axis=0), marker='o', linewidth=2)
            plt.title('Average Feedback Gate Value per Iteration')
            plt.xlabel('Iteration')
            plt.ylabel('Gate Value')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(output_dir, f"feedback_gates_{timestamp}.png"))
            plt.close()
    
    # Compare standard vs final recurrent phase vectors
    visualize_before_after_comparison(
        standard_phases, recurrent_phases, standard_gated_categories,
        title="Standard vs Recurrent Phases"
    )
    plt.savefig(os.path.join(output_dir, f"standard_vs_recurrent_{timestamp}.png"))
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Recurrent Thalamus-Workspace Demo')
    parser.add_argument('--output_dir', type=str, default='recurrent_results',
                      help='Directory to save visualizations')
    args = parser.parse_args()
    
    # Create synthetic data
    print("Creating synthetic data...")
    batch_size = 2
    num_tokens = 100
    d_model = 128
    
    tokens, categories = create_synthetic_data(
        batch_size=batch_size,
        num_tokens=num_tokens,
        d_model=d_model,
        num_categories=4
    )
    
    # Create task IDs
    task_ids = torch.zeros(batch_size, dtype=torch.long)
    
    # Compare standard vs recurrent thalamus-workspace
    compare_models(tokens, categories, task_ids, args.output_dir)


if __name__ == "__main__":
    main()
