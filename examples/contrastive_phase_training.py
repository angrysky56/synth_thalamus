"""
Contrastive Phase Training Demo

This script demonstrates how to train a model with the Synthetic Thalamus
using contrastive learning to improve phase vector quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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

from core.train_utils import ContrastivePhaseTrainer
from core.thalamus import SyntheticThalamus
from core.enhanced_workspace import EnhancedWorkspace
from core.phase_generator import PhaseAnalyzer
from core.visualization import plot_phase_similarity_matrix, plot_intra_inter_similarity


class SimpleModel(nn.Module):
    """
    Simple model with Synthetic Thalamus and Enhanced Workspace.
    """
    def __init__(self, d_model=128, n_heads=4, k=16, phase_dim=16,
                 task_dim=64, num_tasks=10, hidden_dim=256, output_dim=10,
                 phase_config=None):
        super().__init__()
        
        # Default phase generator configuration
        if phase_config is None:
            phase_config = {
                'hidden_dims': [128, 64],
                'activation': 'gelu',
                'phase_diversity': 2.0,
                'use_layer_norm': True
            }
        
        # Synthetic Thalamus
        self.thalamus = SyntheticThalamus(
            d_model=d_model,
            n_heads=n_heads,
            k=k,
            phase_dim=phase_dim,
            task_dim=task_dim,
            num_tasks=num_tasks,
            **phase_config
        )
        
        # Enhanced Workspace
        self.workspace = EnhancedWorkspace(
            input_dim=d_model,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            nhead=n_heads,
            phase_dim=phase_dim,
            num_layers=2
        )
    
    def forward(self, x, task_ids=None):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor [B, N, D]
            task_ids: Optional task IDs [B]
            
        Returns:
            output: Output tensor [B, output_dim]
        """
        # Default task ID if not provided
        if task_ids is None:
            task_ids = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Pass through thalamus
        gated, _ = self.thalamus(x, task_ids)
        
        # Pass through workspace
        output, _ = self.workspace(gated)
        
        return output


def create_synthetic_data(num_samples=1000, num_tokens=32, d_model=128, 
                        num_categories=4, num_classes=10, seed=42):
    """
    Create synthetic data for training and evaluation.
    
    Args:
        num_samples: Number of samples
        num_tokens: Number of tokens per sample
        d_model: Dimensionality of token embeddings
        num_categories: Number of semantic categories
        num_classes: Number of output classes
        seed: Random seed for reproducibility
        
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
    """
    torch.manual_seed(seed)
    
    # Create token embeddings
    tokens = torch.randn(num_samples, num_tokens, d_model)
    
    # Create category IDs (evenly distributed)
    categories = torch.zeros(num_samples, num_tokens, dtype=torch.long)
    tokens_per_category = num_tokens // num_categories
    
    for i in range(num_samples):
        for c in range(num_categories):
            start_idx = c * tokens_per_category
            end_idx = (c + 1) * tokens_per_category if c < num_categories - 1 else num_tokens
            
            # Add category bias to embeddings
            category_direction = torch.randn(d_model)
            tokens[i, start_idx:end_idx] += category_direction.unsqueeze(0) * 1.0
            
            # Assign category IDs
            categories[i, start_idx:end_idx] = c
    
    # Create task IDs (one per sample)
    task_ids = torch.randint(0, 5, (num_samples,))
    
    # Create target classes (related to task and primary category)
    targets = torch.zeros(num_samples, dtype=torch.long)
    for i in range(num_samples):
        # Primary category is the one with the most tokens
        primary_category = torch.bincount(categories[i]).argmax().item()
        
        # Target depends on task ID and primary category
        targets[i] = (task_ids[i] * 2 + primary_category) % num_classes
    
    # Split into train and validation sets (80% train, 20% val)
    train_size = int(0.8 * num_samples)
    
    train_tokens = tokens[:train_size]
    train_categories = categories[:train_size]
    train_task_ids = task_ids[:train_size]
    train_targets = targets[:train_size]
    
    val_tokens = tokens[train_size:]
    val_categories = categories[train_size:]
    val_task_ids = task_ids[train_size:]
    val_targets = targets[train_size:]
    
    # Create datasets and data loaders
    train_dataset = TensorDataset(train_tokens, train_targets, train_task_ids, train_categories)
    val_dataset = TensorDataset(val_tokens, val_targets, val_task_ids, val_categories)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader


def visualize_phases(model, tokens, categories, output_dir, label=""):
    """
    Visualize phase vectors and similarity matrices.
    
    Args:
        model: Model with thalamus and phase generator
        tokens: Input tensor [B, N, D]
        categories: Category IDs [B, N]
        output_dir: Directory to save visualizations
        label: Label for the visualization files
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate phase vectors
    with torch.no_grad():
        # Get task IDs (zeros for simplicity)
        task_ids = torch.zeros(tokens.size(0), dtype=torch.long)
        
        # Pass through thalamus
        gated = model.thalamus(tokens, task_ids)
        
        # Extract phase vectors (assuming d_model is 128)
        content_dim = tokens.size(-1)
        phases = gated[..., content_dim:]
    
    # Create visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Phase similarity matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_phase_similarity_matrix(phases[0], ax=ax, 
                               title=f"Phase Similarity Matrix ({label})")
    fig.savefig(os.path.join(output_dir, f"phase_similarity_{label}_{timestamp}.png"))
    plt.close(fig)
    
    # 2. Intra vs Inter category similarities
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_intra_inter_similarity(
        phases, categories, ax=ax,
        title=f"Category Similarity Analysis ({label})"
    )
    fig.savefig(os.path.join(output_dir, f"category_similarity_{label}_{timestamp}.png"))
    plt.close(fig)
    
    # 3. Visualize individual phase vectors
    fig, ax = plt.subplots(figsize=(12, 8))
    for i in range(min(phases.shape[1], 16)):
        category = categories[0, i].item()
        ax.plot(range(phases.shape[2]), phases[0, i].cpu().numpy(), 
               marker='o', alpha=0.7, 
               label=f"Category {category}" if i < 8 else None)
    
    ax.set_title(f"Phase Vectors ({label})")
    ax.set_xlabel("Phase Dimension")
    ax.set_ylabel("Phase Value")
    ax.grid(True, linestyle='--', alpha=0.5)
    if phases.shape[1] <= 8:
        ax.legend()
    
    fig.savefig(os.path.join(output_dir, f"phase_vectors_{label}_{timestamp}.png"))
    plt.close(fig)
    
    print(f"Phase visualizations ({label}) saved to {output_dir}")
    
    # Compute and return metrics for comparison
    analyzer = PhaseAnalyzer()
    metrics = analyzer.compute_category_similarities(phases, categories)
    
    return {
        'intra_similarity': metrics['intra_similarity'].mean().item(),
        'inter_similarity': metrics['inter_similarity'].mean().item(),
        'contrast': metrics['contrast'].mean().item()
    }


def train_with_contrastive_learning(output_dir, contrastive_weight=0.5):
    """
    Train a model with contrastive learning.
    
    Args:
        output_dir: Directory to save model checkpoints and visualizations
        contrastive_weight: Weight of the contrastive loss term
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Parameters
    d_model = 128
    n_heads = 4
    k = 16
    phase_dim = 16
    task_dim = 64
    num_tasks = 10
    hidden_dim = 256
    num_classes = 10
    
    # Phase generator configuration
    phase_config = {
        'hidden_dims': [256, 128, 64],
        'activation': 'gelu',
        'phase_diversity': 2.0,
        'use_layer_norm': True
    }
    
    # Create model
    model = SimpleModel(
        d_model=d_model,
        n_heads=n_heads,
        k=k,
        phase_dim=phase_dim,
        task_dim=task_dim,
        num_tasks=num_tasks,
        hidden_dim=hidden_dim,
        output_dim=num_classes,
        phase_config=phase_config
    )
    
    # Create synthetic data
    train_loader, val_loader = create_synthetic_data(
        num_samples=1000,
        num_tokens=32,
        d_model=d_model,
        num_categories=4,
        num_classes=num_classes
    )
    
    # Extract a batch for visualization
    for batch in val_loader:
        vis_tokens = batch[0]
        vis_categories = batch[3]
        break
    
    # Visualize phases before training
    print("\nVisualizing phases before training...")
    before_metrics = visualize_phases(
        model=model,
        tokens=vis_tokens,
        categories=vis_categories,
        output_dir=output_dir,
        label="before_training"
    )
    
    # Create contrastive trainer
    trainer = ContrastivePhaseTrainer(
        model=model,
        contrastive_weight=contrastive_weight,
        temperature=0.5
    )
    
    # Train the model
    print("\nTraining with contrastive loss...")
    metrics = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=10,
        learning_rate=0.001,
        weight_decay=1e-5,
        task_id_key=2,  # Index of task_ids in batch
        category_key=3,  # Index of categories in batch
        save_dir=output_dir,
        save_freq=10
    )
    
    # Visualize phases after training
    print("\nVisualizing phases after training...")
    after_metrics = visualize_phases(
        model=model,
        tokens=vis_tokens,
        categories=vis_categories,
        output_dir=output_dir,
        label="after_training"
    )
    
    # Compare metrics before and after training
    print("\nCategory Similarity Metrics:")
    print("                   Before      After     Change")
    print("-------------------------------------------------")
    print(f"Intra-Similarity:  {before_metrics['intra_similarity']:.4f}      {after_metrics['intra_similarity']:.4f}      {after_metrics['intra_similarity'] - before_metrics['intra_similarity']:.4f}")
    print(f"Inter-Similarity:  {before_metrics['inter_similarity']:.4f}      {after_metrics['inter_similarity']:.4f}      {after_metrics['inter_similarity'] - before_metrics['inter_similarity']:.4f}")
    print(f"Contrast:          {before_metrics['contrast']:.4f}      {after_metrics['contrast']:.4f}      {after_metrics['contrast'] - before_metrics['contrast']:.4f}")
    
    # Plot training metrics
    epochs = range(1, len(metrics['train_loss']) + 1)
    
    # Training and validation loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(epochs, metrics['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, metrics['val_loss'], 'r-', label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Task loss and contrastive loss
    ax2.plot(epochs, metrics['train_task_loss'], 'g-', label='Task Loss')
    ax2.plot(epochs, metrics['train_contrastive_loss'], 'm-', label='Contrastive Loss')
    ax2.set_title('Task and Contrastive Loss Components')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(output_dir, f"training_metrics_{timestamp}.png"))
    plt.close(fig)
    
    print(f"Training metrics plot saved to {output_dir}")
    
    # Save final model
    trainer.save_model(os.path.join(output_dir, f"final_model_{timestamp}.pt"))
    
    return model, metrics


def visualize_attention_patterns(model, tokens, categories, output_dir):
    """
    Visualize attention patterns in the enhanced workspace.
    
    Args:
        model: Model with thalamus and enhanced workspace
        tokens: Input tensor [B, N, D]
        categories: Category IDs [B, N]
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate attention patterns
    with torch.no_grad():
        # Get task IDs (zeros for simplicity)
        task_ids = torch.zeros(tokens.size(0), dtype=torch.long)
        
        # Pass through thalamus
        gated = model.thalamus(tokens, task_ids)
        
        # Pass through workspace to get attention weights
        outputs, _ = model.workspace(gated)
        attention_weights = model.workspace.attention_weights
    
    # Create visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Attention patterns for each layer
    num_layers = len(attention_weights)
    fig, axes = plt.subplots(1, num_layers, figsize=(num_layers * 6, 5))
    
    if num_layers == 1:
        axes = [axes]
    
    for i, attn in enumerate(attention_weights):
        # Average over heads and take first batch item
        attn_avg = attn.mean(dim=1)[0].cpu().numpy()
        
        # Plot attention heatmap
        im = axes[i].imshow(attn_avg, cmap='viridis')
        plt.colorbar(im, ax=axes[i], label='Attention Weight')
        axes[i].set_title(f'Layer {i+1} Attention Pattern')
        axes[i].set_xlabel('Target Token')
        axes[i].set_ylabel('Source Token')
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"attention_patterns_{timestamp}.png"))
    plt.close(fig)
    
    # 2. Attention vs. categories
    # Create a matrix showing how attention aligns with categories
    fig, axes = plt.subplots(1, num_layers, figsize=(num_layers * 6, 5))
    
    if num_layers == 1:
        axes = [axes]
    
    # Get category matrix (1 where same category, 0 where different)
    categories_np = categories[0].cpu().numpy()
    category_matrix = np.zeros((categories_np.shape[0], categories_np.shape[0]))
    for i in range(category_matrix.shape[0]):
        for j in range(category_matrix.shape[1]):
            category_matrix[i, j] = 1 if categories_np[i] == categories_np[j] else 0
    
    for i, attn in enumerate(attention_weights):
        # Average over heads and take first batch item
        attn_avg = attn.mean(dim=1)[0].cpu().numpy()
        
        # Calculate category-attention alignment
        same_cat_attn = attn_avg * category_matrix
        diff_cat_attn = attn_avg * (1 - category_matrix)
        
        same_cat_mean = same_cat_attn.sum() / category_matrix.sum() if category_matrix.sum() > 0 else 0
        diff_cat_mean = diff_cat_attn.sum() / (1 - category_matrix).sum() if (1 - category_matrix).sum() > 0 else 0
        
        # Create side-by-side bar chart
        axes[i].bar([0, 1], [same_cat_mean, diff_cat_mean], alpha=0.7,
                  tick_label=['Same Category', 'Different Category'])
        axes[i].set_title(f'Layer {i+1} Attention by Category')
        axes[i].set_ylabel('Average Attention Weight')
        axes[i].grid(True, linestyle='--', alpha=0.7)
        
        # Add value labels
        axes[i].text(0, same_cat_mean, f"{same_cat_mean:.4f}", ha='center', va='bottom')
        axes[i].text(1, diff_cat_mean, f"{diff_cat_mean:.4f}", ha='center', va='bottom')
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"attention_by_category_{timestamp}.png"))
    plt.close(fig)
    
    print(f"Attention pattern visualizations saved to {output_dir}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Contrastive Phase Training Demo')
    parser.add_argument('--output_dir', type=str, default='contrastive_results',
                      help='Directory to save output visualizations and models')
    parser.add_argument('--contrastive_weight', type=float, default=0.5,
                      help='Weight of the contrastive loss term')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Train model with contrastive learning
    print("\nTraining model with contrastive learning...")
    model, metrics = train_with_contrastive_learning(
        output_dir=args.output_dir,
        contrastive_weight=args.contrastive_weight
    )
    
    # Create synthetic data for attention visualization
    print("\nCreating synthetic data for attention visualization...")
    tokens, categories = create_synthetic_data(
        num_samples=2,
        num_tokens=32,
        d_model=128,
        num_categories=4
    )[0].dataset.tensors[:2]
    
    # Visualize attention patterns
    print("\nVisualizing attention patterns...")
    visualize_attention_patterns(
        model=model,
        tokens=tokens,
        categories=categories,
        output_dir=args.output_dir
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
