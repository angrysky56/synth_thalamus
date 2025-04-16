"""
Visualization tools for phase similarity analysis.

This module provides functions to visualize and analyze phase vectors,
attention patterns, and phase similarity metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap


def plot_phase_similarity_matrix(phase_vectors, ax=None, title="Phase Similarity Matrix"):
    """
    Plot a heatmap of phase vector similarities.
    
    Args:
        phase_vectors: Tensor of shape [B, k, phase_dim] or [k, phase_dim]
        ax: Matplotlib axis for plotting (optional)
        title: Plot title
        
    Returns:
        ax: The matplotlib axis
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    
    # Handle different input shapes
    if phase_vectors.dim() == 3:
        # Take the first batch item
        phase_vectors = phase_vectors[0]
    
    # Compute similarity matrix
    phase_norm = F.normalize(phase_vectors, p=2, dim=1)
    similarity = torch.mm(phase_norm, phase_norm.t()).cpu().numpy()
    
    # Zero out diagonal for better visualization
    np.fill_diagonal(similarity, 0)
    
    # Plot the similarity matrix
    im = ax.imshow(similarity, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    
    ax.set_title(title)
    ax.set_xlabel('Token Index')
    ax.set_ylabel('Token Index')
    
    return ax


def plot_attention_comparison(phase_vectors, attn_weights, phase_scale_values=None, 
                             fig=None, title="Phase Similarity vs Attention"):
    """
    Plot a comparison of phase similarity and attention patterns.
    
    Args:
        phase_vectors: Tensor of shape [B, k, phase_dim] or [k, phase_dim]
        attn_weights: List of attention weight tensors [B, nhead, k, k]
        phase_scale_values: List of phase scale values used (optional)
        fig: Matplotlib figure for plotting (optional)
        title: Main title for the figure
        
    Returns:
        fig: The matplotlib figure
    """
    if fig is None:
        fig = plt.figure(figsize=(12, 8))
    
    # Handle different input shapes
    if phase_vectors.dim() == 3:
        # Take the first batch item
        phase_vectors = phase_vectors[0]
    
    # Compute phase similarity matrix
    phase_norm = F.normalize(phase_vectors, p=2, dim=1)
    phase_sim = torch.mm(phase_norm, phase_norm.t()).cpu().numpy()
    
    # Zero out diagonal for better visualization
    phase_sim_viz = phase_sim.copy()
    np.fill_diagonal(phase_sim_viz, 0)
    
    # Number of attention matrices to visualize
    num_attn = len(attn_weights)
    
    # Create grid of plots
    gs = fig.add_gridspec(2, num_attn + 1)
    
    # Plot phase similarity
    ax_phase = fig.add_subplot(gs[0, 0])
    im = ax_phase.imshow(phase_sim_viz, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax_phase, label='Similarity')
    ax_phase.set_title("Phase Similarity")
    ax_phase.set_xlabel("Token Index")
    ax_phase.set_ylabel("Token Index")
    
    # Plot phase vectors
    ax_vectors = fig.add_subplot(gs[1, 0])
    for i in range(phase_vectors.shape[0]):
        ax_vectors.plot(range(phase_vectors.shape[1]), 
                       phase_vectors[i].cpu().numpy(), 
                       marker='o', 
                       label=f"Token {i}" if i < 5 else None)
    
    ax_vectors.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax_vectors.grid(True, linestyle='--', alpha=0.5)
    ax_vectors.set_xlabel("Phase Dimension")
    ax_vectors.set_ylabel("Phase Value")
    ax_vectors.set_title("Phase Vectors")
    if phase_vectors.shape[0] <= 5:
        ax_vectors.legend()
    
    # Plot attention weights for each provided attention matrix
    for i, attn in enumerate(attn_weights):
        # Average over heads and take first batch item
        if attn.dim() == 4:
            attn_avg = attn.mean(dim=1)[0].cpu().numpy()
        else:
            attn_avg = attn.cpu().numpy()
        
        # Plot attention weights
        ax_attn = fig.add_subplot(gs[0, i + 1])
        im = ax_attn.imshow(attn_avg, cmap='viridis')
        plt.colorbar(im, ax=ax_attn, label='Weight')
        
        # Set title based on phase scale if provided
        if phase_scale_values is not None:
            ax_attn.set_title(f"Attention (Scale={phase_scale_values[i]:.1f})")
        else:
            ax_attn.set_title(f"Attention {i+1}")
            
        ax_attn.set_xlabel("Target Token")
        ax_attn.set_ylabel("Source Token")
        
        # Plot difference between attention and phase similarity
        diff = attn_avg - phase_sim
        ax_diff = fig.add_subplot(gs[1, i + 1])
        
        # Create a diverging colormap with white at zero
        cmap = LinearSegmentedColormap.from_list(
            'BrBG_r', [(0, '#053061'), (0.5, '#f7f7f7'), (1, '#67001f')])
        
        im = ax_diff.imshow(diff, cmap=cmap)
        plt.colorbar(im, ax=ax_diff, label='Difference')
        ax_diff.set_title("Attention - Phase Sim")
        ax_diff.set_xlabel("Token Index")
        ax_diff.set_ylabel("Token Index")
    
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    
    return fig


def visualize_phase_diversity(phase_generators, inputs, 
                              diversity_values=None, 
                              fig=None, 
                              title="Phase Diversity Comparison"):
    """
    Visualize phase vectors generated with different diversity parameters.
    
    Args:
        phase_generators: List of phase generator models
        inputs: Tensor of shape [B, k, d_model] to feed into the generators
        diversity_values: List of diversity parameter values (optional)
        fig: Matplotlib figure for plotting (optional)
        title: Main title for the figure
        
    Returns:
        fig: The matplotlib figure
    """
    if fig is None:
        fig = plt.figure(figsize=(15, 10))
    
    num_gens = len(phase_generators)
    
    # Create grid of plots: top row for phase vectors, bottom row for similarity matrices
    gs = fig.add_gridspec(2, num_gens)
    
    # Evaluate each phase generator
    for i, generator in enumerate(phase_generators):
        with torch.no_grad():
            # Generate phase vectors
            phase = generator(inputs)
            
            # Take first batch item
            phase_single = phase[0].cpu()
            
            # Plot phase vectors
            ax_phase = fig.add_subplot(gs[0, i])
            for j in range(phase_single.shape[0]):
                ax_phase.plot(range(phase_single.shape[1]), 
                             phase_single[j].numpy(), 
                             marker='o', 
                             alpha=0.7,
                             label=f"Token {j}" if j < 5 else None)
            
            ax_phase.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax_phase.grid(True, linestyle='--', alpha=0.5)
            ax_phase.set_xlabel("Phase Dimension")
            ax_phase.set_ylabel("Phase Value")
            
            # Set title based on diversity value if provided
            if diversity_values is not None:
                ax_phase.set_title(f"Diversity = {diversity_values[i]:.1f}")
            else:
                ax_phase.set_title(f"Generator {i+1}")
                
            if phase_single.shape[0] <= 5:
                ax_phase.legend()
            
            # Plot similarity matrix
            ax_sim = fig.add_subplot(gs[1, i])
            plot_phase_similarity_matrix(phase_single, ax=ax_sim, 
                                       title="Phase Similarity Matrix")
    
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    
    return fig


def plot_intra_inter_similarity(phase_vectors, categories, 
                              ax=None, title="Category Similarity Analysis"):
    """
    Plot intra-category vs inter-category phase similarities.
    
    Args:
        phase_vectors: Tensor of shape [B, k, phase_dim] or [k, phase_dim]
        categories: Tensor of shape [B, k] or [k] containing category IDs
        ax: Matplotlib axis for plotting (optional)
        title: Plot title
        
    Returns:
        ax: The matplotlib axis
        metrics: Dictionary of similarity metrics
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    
    # Handle different input shapes
    if phase_vectors.dim() == 3:
        # Take the first batch item
        phase_vectors = phase_vectors[0]
        if categories.dim() == 2:
            categories = categories[0]
    
    # Print debug information
    print(f"Phase vectors shape: {phase_vectors.shape}")
    print(f"Categories shape: {categories.shape}")
    
    # Compute similarity matrix
    phase_norm = F.normalize(phase_vectors, p=2, dim=1)
    similarity = torch.mm(phase_norm, phase_norm.t()).cpu()
    
    # Convert tensors to cpu numpy arrays for safer processing
    similarity_np = similarity.numpy()
    categories_np = categories.cpu().numpy()
    
    print(f"Similarity matrix shape: {similarity_np.shape}")
    print(f"Unique categories: {np.unique(categories_np)}")
    
    # Initialize metrics with safe defaults
    intra_mean = 0
    inter_mean = 0
    contrast = 0
    category_sizes = []
    
    # Skip detailed category analysis if shapes don't match
    # This is a simplification to avoid the index error
    if similarity_np.shape[0] != similarity_np.shape[1]:
        print("Warning: Similarity matrix is not square. Skipping detailed analysis.")
    else:
        try:
            # Get unique categories
            unique_categories = np.unique(categories_np)
            
            # Initialize lists for metrics
            intra_sims = []
            inter_sims = []
            
            # Create a simplified robust approach with careful bounds checking
            for cat in unique_categories:
                cat_indices = np.where(categories_np == cat)[0]
                category_sizes.append(len(cat_indices))
                print(f"Category {cat} has {len(cat_indices)} tokens")
                
                if len(cat_indices) <= 1:
                    # Skip categories with only one token
                    continue
                
                # Compute intra-category similarity (excluding self-comparisons)
                intra_cat_sims = []
                for i, idx1 in enumerate(cat_indices):
                    for j, idx2 in enumerate(cat_indices):
                        if i != j and idx1 < similarity_np.shape[0] and idx2 < similarity_np.shape[1]:  
                            # Ensure indices are in bounds
                            intra_cat_sims.append(similarity_np[idx1, idx2])
                
                if intra_cat_sims:
                    intra_sim = np.mean(intra_cat_sims)
                    intra_sims.append(intra_sim)
                
                # Compute inter-category similarities
                inter_cat_sims = []
                for other_cat in unique_categories:
                    if cat == other_cat:
                        continue
                        
                    other_indices = np.where(categories_np == other_cat)[0]
                    if len(other_indices) == 0:
                        continue
                    
                    # Use explicit loops with bounds checking
                    cat_other_sims = []
                    for idx1 in cat_indices:
                        if idx1 >= similarity_np.shape[0]:
                            continue
                        for idx2 in other_indices:
                            if idx2 >= similarity_np.shape[1]:
                                continue
                            cat_other_sims.append(similarity_np[idx1, idx2])
                    
                    if cat_other_sims:
                        inter_cat_sims.append(np.mean(cat_other_sims))
                    
                if inter_cat_sims:
                    inter_sims.append(np.mean(inter_cat_sims))
            
            # Compute overall metrics
            intra_mean = np.mean(intra_sims) if intra_sims else 0
            inter_mean = np.mean(inter_sims) if inter_sims else 0
            contrast = intra_mean - inter_mean
        except Exception as e:
            print(f"Error in category analysis: {e}")
    
    # Compute overall metrics
    intra_mean = np.mean(intra_sims) if intra_sims else 0
    inter_mean = np.mean(inter_sims) if inter_sims else 0
    contrast = intra_mean - inter_mean
    
    # Plot metrics
    bar_width = 0.35
    cat_labels = ['Intra-Category', 'Inter-Category']
    values = [intra_mean, inter_mean]
    
    ax.bar(cat_labels, values, bar_width, alpha=0.8)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Add value labels on top of bars
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.3f}", ha='center')
    
    # Add contrast value with safe max value calculation
    max_value = max(values) if values and max(values) > 0 else 0.1
    ax.text(0.5, max_value + 0.1, f"Contrast: {contrast:.3f}", 
           ha='center', fontweight='bold')
    
    # Set reasonable y-limits with safer calculations
    min_value = min(values) if values and min(values) < 0 else -0.1
    max_value = max(values) if values and max(values) > 0 else 0.2
    ax.set_ylim(min(min_value - 0.1, -0.1), max(max_value + 0.2, 0.2))
    
    ax.set_title(title)
    ax.set_ylabel('Average Cosine Similarity')
    
    # Create metrics dictionary
    metrics = {
        'intra_similarity': intra_mean,
        'inter_similarity': inter_mean,
        'contrast': contrast,
        'category_sizes': category_sizes
    }
    
    return ax, metrics
    
    ax.set_title(title)
    ax.set_ylabel('Average Cosine Similarity')
    
    # Create metrics dictionary
    metrics = {
        'intra_similarity': intra_mean,
        'inter_similarity': inter_mean,
        'contrast': contrast,
        'category_sizes': category_sizes
    }
    
    return ax, metrics
