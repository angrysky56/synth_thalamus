"""
Enhanced visualization tools for the Synthetic Thalamus

This module extends the basic visualization capabilities with more advanced
analysis tools specifically designed for contrastive learning evaluation.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def visualize_loss_components(metrics, figsize=(12, 8), title="Training Loss Components"):
    """
    Visualize task loss and contrastive loss components over training.
    
    Args:
        metrics: Dictionary of training metrics
        figsize: Figure size
        title: Plot title
        
    Returns:
        fig: Matplotlib figure
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    epochs = range(1, len(metrics['train_loss']) + 1)
    
    # Overall loss
    ax1.plot(epochs, metrics['train_loss'], 'b-', label='Total Loss')
    ax1.set_title('Total Training Loss')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Task vs Contrastive loss
    ax2.plot(epochs, metrics['train_task_loss'], 'g-', label='Task Loss')
    ax2.plot(epochs, metrics['train_contrastive_loss'], 'm-', label='Contrastive Loss')
    ax2.set_title('Task vs Contrastive Loss')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Validation metrics
    if 'val_loss' in metrics and 'val_accuracy' in metrics:
        ax3.plot(epochs, metrics['val_loss'], 'r-', label='Val Loss')
        ax3.set_title('Validation Metrics')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # Add accuracy on second y-axis
        ax3_acc = ax3.twinx()
        ax3_acc.plot(epochs, metrics['val_accuracy'], 'c-', label='Val Accuracy')
        ax3_acc.set_ylabel('Accuracy')
        ax3_acc.set_ylim(0, 1.0)
        
        # Combine legends
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_acc.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    return fig


def visualize_hyperparameters(metrics, figsize=(12, 5), title="Hyperparameter Evolution"):
    """
    Visualize evolution of hyperparameters like temperature and contrastive weight.
    
    Args:
        metrics: Dictionary of training metrics
        figsize: Figure size
        title: Plot title
        
    Returns:
        fig: Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(metrics['train_loss']) + 1)
    
    # Temperature
    if 'temperature' in metrics:
        ax1.plot(epochs, metrics['temperature'], 'r-', marker='o')
        ax1.set_title('Temperature Evolution')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Temperature')
        ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Contrastive weight
    if 'contrastive_weight' in metrics:
        ax2.plot(epochs, metrics['contrastive_weight'], 'b-', marker='o')
        ax2.set_title('Contrastive Weight Evolution')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Weight')
        ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig


def visualize_phase_space(phases, categories, method='tsne', figsize=(10, 8), 
                         title="Phase Vector Visualization"):
    """
    Visualize phase vectors in a 2D projection using t-SNE or PCA.
    
    Args:
        phases: Tensor of shape [B, k, phase_dim] or [k, phase_dim]
        categories: Tensor of shape [B, k] or [k] containing category IDs
        method: Dimensionality reduction method ('tsne' or 'pca')
        figsize: Figure size
        title: Plot title
        
    Returns:
        fig: Matplotlib figure
    """
    # Convert to numpy for scikit-learn
    if phases.dim() == 3:
        # Take the first batch
        phases = phases[0]
        if categories.dim() == 2:
            categories = categories[0]
    
    phase_np = phases.detach().cpu().numpy()
    categories_np = categories.detach().cpu().numpy()
    
    # Apply dimensionality reduction
    if method.lower() == 'tsne':
        model = TSNE(n_components=2, perplexity=min(30, phases.size(0)//2), random_state=42)
        reduced = model.fit_transform(phase_np)
    else:  # pca
        model = PCA(n_components=2, random_state=42)
        reduced = model.fit_transform(phase_np)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique categories for coloring
    unique_categories = np.unique(categories_np)
    
    # Plot each category with a different color
    for category in unique_categories:
        mask = categories_np == category
        ax.scatter(reduced[mask, 0], reduced[mask, 1], label=f'Category {category}',
                 alpha=0.8, edgecolors='w', linewidth=0.5)
    
    ax.set_title(f"{title} ({method.upper()})")
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig


def visualize_category_similarity_matrix(phases, categories, figsize=(10, 8),
                                       title="Category-wise Phase Similarity"):
    """
    Visualize average similarity between phases grouped by category.
    
    Args:
        phases: Tensor of shape [B, k, phase_dim] or [k, phase_dim]
        categories: Tensor of shape [B, k] or [k] containing category IDs
        figsize: Figure size
        title: Plot title
        
    Returns:
        fig: Matplotlib figure
    """
    # Handle different input shapes
    if phases.dim() == 3:
        # Take the first batch
        phases = phases[0]
        if categories.dim() == 2:
            categories = categories[0]
    
    # Get unique categories
    unique_categories = torch.unique(categories)
    num_categories = len(unique_categories)
    
    # Create category-wise similarity matrix
    category_sim = torch.zeros((num_categories, num_categories))
    
    # Normalize phases for cosine similarity
    phases_norm = F.normalize(phases, p=2, dim=1)
    
    # Compute token-wise similarity
    token_sim = torch.mm(phases_norm, phases_norm.t())
    
    # Compute category-wise similarity by averaging token similarities
    for i, cat_i in enumerate(unique_categories):
        for j, cat_j in enumerate(unique_categories):
            # Get indices for each category
            indices_i = torch.where(categories == cat_i)[0]
            indices_j = torch.where(categories == cat_j)[0]
            
            # Get similarities between these indices
            if i == j:  # Exclude self-similarities for intra-category
                # Create a mask to exclude self-similarities
                mask = torch.ones((len(indices_i), len(indices_j)), dtype=torch.bool)
                for k in range(len(indices_i)):
                    mask[k, indices_j == indices_i[k]] = False
                
                # Extract similarities with mask
                sims = token_sim[indices_i][:, indices_j]
                if mask.any():
                    sims = sims[mask]
                    category_sim[i, j] = sims.mean() if len(sims) > 0 else 0
                else:
                    category_sim[i, j] = 0
            else:
                # Get all similarities between different categories
                sims = token_sim[indices_i][:, indices_j]
                category_sim[i, j] = sims.mean()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(category_sim.cpu().numpy(), cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label='Average Cosine Similarity')
    
    # Set up axes
    ax.set_title(title)
    ax.set_xlabel('Category')
    ax.set_ylabel('Category')
    
    # Add category labels
    ax.set_xticks(np.arange(num_categories))
    ax.set_yticks(np.arange(num_categories))
    ax.set_xticklabels([f'Cat {c.item()}' for c in unique_categories])
    ax.set_yticklabels([f'Cat {c.item()}' for c in unique_categories])
    
    # Add similarity values
    for i in range(num_categories):
        for j in range(num_categories):
            value = category_sim[i, j].item()
            text_color = 'white' if abs(value) > 0.5 else 'black'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', color=text_color)
    
    plt.tight_layout()
    return fig


def visualize_before_after_comparison(before_phases, after_phases, categories, 
                                    figsize=(15, 10), title="Before vs After Training"):
    """
    Visualize phase vectors before and after training.
    
    Args:
        before_phases: Tensor of phases before training [B, k, phase_dim] or [k, phase_dim]
        after_phases: Tensor of phases after training [B, k, phase_dim] or [k, phase_dim]
        categories: Tensor of category IDs [B, k] or [k]
        figsize: Figure size
        title: Plot title
        
    Returns:
        fig: Matplotlib figure
    """
    # Handle different input shapes
    if before_phases.dim() == 3:
        # Take first batch
        before_phases = before_phases[0]
        after_phases = after_phases[0]
        if categories.dim() == 2:
            categories = categories[0]
    
    # Create figure with 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Before t-SNE visualization
    if before_phases.shape[0] >= 8:  # Need enough points for t-SNE
        before_np = before_phases.detach().cpu().numpy()
        categories_np = categories.detach().cpu().numpy()
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=min(30, before_phases.size(0)//2), random_state=42)
        before_reduced = tsne.fit_transform(before_np)
        
        # Plot each category with a different color
        unique_categories = np.unique(categories_np)
        for category in unique_categories:
            mask = categories_np == category
            axes[0, 0].scatter(before_reduced[mask, 0], before_reduced[mask, 1], 
                             label=f'Cat {category}', alpha=0.8, edgecolors='w', linewidth=0.5)
        
        axes[0, 0].set_title('Before Training (t-SNE)')
        axes[0, 0].set_xlabel('Component 1')
        axes[0, 0].set_ylabel('Component 2')
        axes[0, 0].legend()
        axes[0, 0].grid(True, linestyle='--', alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, "Not enough data points for t-SNE", 
                      ha='center', va='center', transform=axes[0, 0].transAxes)
    
    # 2. After t-SNE visualization
    if after_phases.shape[0] >= 8:  # Need enough points for t-SNE
        after_np = after_phases.detach().cpu().numpy()
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=min(30, after_phases.size(0)//2), random_state=42)
        after_reduced = tsne.fit_transform(after_np)
        
        # Plot each category with a different color
        for category in unique_categories:
            mask = categories_np == category
            axes[0, 1].scatter(after_reduced[mask, 0], after_reduced[mask, 1], 
                             label=f'Cat {category}', alpha=0.8, edgecolors='w', linewidth=0.5)
        
        axes[0, 1].set_title('After Training (t-SNE)')
        axes[0, 1].set_xlabel('Component 1')
        axes[0, 1].set_ylabel('Component 2')
        axes[0, 1].legend()
        axes[0, 1].grid(True, linestyle='--', alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, "Not enough data points for t-SNE", 
                      ha='center', va='center', transform=axes[0, 1].transAxes)
    
    # 3. Before similarity matrix
    before_norm = F.normalize(before_phases, p=2, dim=1)
    before_sim = torch.mm(before_norm, before_norm.t()).cpu().numpy()
    
    # Zero out diagonal for better visualization
    np.fill_diagonal(before_sim, 0)
    
    im = axes[1, 0].imshow(before_sim, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar(im, ax=axes[1, 0], label='Cosine Similarity')
    axes[1, 0].set_title('Before Training (Similarity)')
    axes[1, 0].set_xlabel('Token Index')
    axes[1, 0].set_ylabel('Token Index')
    
    # 4. After similarity matrix
    after_norm = F.normalize(after_phases, p=2, dim=1)
    after_sim = torch.mm(after_norm, after_norm.t()).cpu().numpy()
    
    # Zero out diagonal for better visualization
    np.fill_diagonal(after_sim, 0)
    
    im = axes[1, 1].imshow(after_sim, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar(im, ax=axes[1, 1], label='Cosine Similarity')
    axes[1, 1].set_title('After Training (Similarity)')
    axes[1, 1].set_xlabel('Token Index')
    axes[1, 1].set_ylabel('Token Index')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig


def visualize_gradient_analysis(grad_stats, figsize=(12, 8), title="Gradient Analysis"):
    """
    Visualize gradient statistics for the phase generator.
    
    Args:
        grad_stats: Dictionary of gradient statistics
        figsize: Figure size
        title: Plot title
        
    Returns:
        fig: Matplotlib figure
    """
    # Extract data from stats
    params = []
    for key in grad_stats.keys():
        if '_norm' in key:
            param = key.replace('_norm', '')
            params.append(param)
    
    # Create figure
    fig, axes = plt.subplots(len(params), 1, figsize=figsize)
    if len(params) == 1:
        axes = [axes]
    
    for i, param in enumerate(params):
        ax = axes[i]
        
        # Plot gradient norm
        steps = range(len(grad_stats[f'{param}_norm']))
        ax.plot(steps, grad_stats[f'{param}_norm'], label='Norm', linewidth=2)
        
        # Plot mean and std if available
        if f'{param}_mean' in grad_stats:
            ax.plot(steps, grad_stats[f'{param}_mean'], label='Mean', alpha=0.7)
        if f'{param}_std' in grad_stats:
            ax.plot(steps, grad_stats[f'{param}_std'], label='Std', alpha=0.7)
        
        # Mark vanishing/exploding gradients if detected
        if f'{param}_vanishing' in grad_stats and grad_stats[f'{param}_vanishing']:
            for step in grad_stats[f'{param}_vanishing']:
                ax.axvline(x=step, color='blue', linestyle='--', alpha=0.5)
                ax.text(step, ax.get_ylim()[1] * 0.9, 'Vanishing', rotation=90, alpha=0.7)
        
        if f'{param}_exploding' in grad_stats and grad_stats[f'{param}_exploding']:
            for step in grad_stats[f'{param}_exploding']:
                ax.axvline(x=step, color='red', linestyle='--', alpha=0.5)
                ax.text(step, ax.get_ylim()[1] * 0.9, 'Exploding', rotation=90, alpha=0.7)
        
        ax.set_title(f'Parameter: {param}')
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig


def visualize_phase_dimensionality(phases, categories, max_dims=None, figsize=(12, 6),
                                 title="Phase Vector Dimensionality Analysis"):
    """
    Analyze the effective dimensionality of phase vectors using PCA.
    
    Args:
        phases: Tensor of shape [B, k, phase_dim] or [k, phase_dim]
        categories: Tensor of shape [B, k] or [k] containing category IDs
        max_dims: Maximum number of dimensions to analyze
        figsize: Figure size
        title: Plot title
        
    Returns:
        fig: Matplotlib figure
    """
    # Handle different input shapes
    if phases.dim() == 3:
        # Take first batch
        phases = phases[0]
        if categories.dim() == 2:
            categories = categories[0]
    
    # Convert to numpy
    phases_np = phases.detach().cpu().numpy()
    categories_np = categories.detach().cpu().numpy()
    
    # Get number of dimensions
    n_dims = phases.shape[1]
    if max_dims is None:
        max_dims = n_dims
    else:
        max_dims = min(max_dims, n_dims)
    
    # Apply PCA
    pca = PCA(n_components=max_dims)
    pca.fit(phases_np)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 1. Explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    ax1.bar(range(1, max_dims + 1), explained_variance, alpha=0.7, label='Explained Variance')
    ax1.step(range(1, max_dims + 1), cumulative_variance, where='mid', 
           label='Cumulative Variance', color='red')
    
    # Mark 90% and 95% variance
    thresholds = [0.9, 0.95]
    for threshold in thresholds:
        try:
            dim_idx = np.where(cumulative_variance >= threshold)[0][0]
            ax1.axhline(y=threshold, color='gray', linestyle='--', alpha=0.5)
            ax1.axvline(x=dim_idx + 1, color='gray', linestyle='--', alpha=0.5)
            ax1.text(dim_idx + 1.1, threshold - 0.03, f'{threshold*100:.0f}% @ dim {dim_idx + 1}')
        except IndexError:
            pass
    
    ax1.set_title('PCA Explained Variance')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_xticks(range(1, max_dims + 1, max(1, max_dims // 10)))
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # 2. Projection onto first two principal components
    transformed = pca.transform(phases_np)
    
    # Plot each category
    unique_categories = np.unique(categories_np)
    for category in unique_categories:
        mask = categories_np == category
        ax2.scatter(transformed[mask, 0], transformed[mask, 1], 
                  label=f'Category {category}', alpha=0.8, edgecolors='w', linewidth=0.5)
    
    ax2.set_title('First Two Principal Components')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig


def visualize_contrastive_loss_impact(categories, all_phases, contrastive_weights,
                                    figsize=(15, 10), title="Impact of Contrastive Loss"):
    """
    Visualize the impact of different contrastive loss weights on phase representations.
    
    Args:
        categories: Tensor of category IDs [B, k] or [k]
        all_phases: List of phase tensors for different contrastive weights
        contrastive_weights: List of contrastive weights corresponding to phase tensors
        figsize: Figure size
        title: Plot title
        
    Returns:
        fig: Matplotlib figure
    """
    # Handle different input shapes
    if categories.dim() == 2:
        categories = categories[0]
    
    num_weights = len(contrastive_weights)
    
    # Compute metrics for each setting
    intra_sims = []
    inter_sims = []
    contrasts = []
    
    for phases in all_phases:
        if phases.dim() == 3:
            phases = phases[0]
        
        # Normalize phases
        phases_norm = F.normalize(phases, p=2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(phases_norm, phases_norm.t())
        
        # Get intra- and inter-category similarities
        intra_sim = 0
        inter_sim = 0
        intra_count = 0
        inter_count = 0
        
        for i in range(len(categories)):
            for j in range(len(categories)):
                if i != j:
                    if categories[i] == categories[j]:
                        intra_sim += sim_matrix[i, j].item()
                        intra_count += 1
                    else:
                        inter_sim += sim_matrix[i, j].item()
                        inter_count += 1
        
        # Compute averages
        intra_avg = intra_sim / max(1, intra_count)
        inter_avg = inter_sim / max(1, inter_count)
        contrast = intra_avg - inter_avg
        
        intra_sims.append(intra_avg)
        inter_sims.append(inter_avg)
        contrasts.append(contrast)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 1. Intra- and inter-category similarities
    weights_labels = [str(w) for w in contrastive_weights]
    x = np.arange(len(weights_labels))
    width = 0.35
    
    ax1.bar(x - width/2, intra_sims, width, label='Intra-category Similarity')
    ax1.bar(x + width/2, inter_sims, width, label='Inter-category Similarity')
    
    ax1.set_title('Category Similarities')
    ax1.set_xlabel('Contrastive Loss Weight')
    ax1.set_ylabel('Average Cosine Similarity')
    ax1.set_xticks(x)
    ax1.set_xticklabels(weights_labels)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Add value labels
    for i, v in enumerate(intra_sims):
        ax1.text(i - width/2, v + 0.01, f"{v:.3f}", ha='center')
    for i, v in enumerate(inter_sims):
        ax1.text(i + width/2, v + 0.01, f"{v:.3f}", ha='center')
    
    # 2. Contrast
    ax2.plot(contrastive_weights, contrasts, marker='o', linewidth=2)
    ax2.set_title('Category Contrast')
    ax2.set_xlabel('Contrastive Loss Weight')
    ax2.set_ylabel('Contrast (Intra - Inter)')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Add value labels
    for i, v in enumerate(contrasts):
        ax2.text(contrastive_weights[i], v + 0.01, f"{v:.3f}", ha='center')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig
