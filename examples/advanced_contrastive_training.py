"""
Advanced Contrastive Learning Demo

This script demonstrates the enhanced contrastive learning features for the Synthetic Thalamus,
including temperature scheduling, adaptive loss weighting, hard negative mining,
and gradient monitoring.
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
from core.contrastive_utils import StratifiedBatchSampler
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
        gated = self.thalamus(x, task_ids)
        
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
        test_tokens: Tensor of shape [B, N, D] for visualization
        test_categories: Tensor of shape [B, N] for visualization
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
    
    # Create datasets
    train_dataset = TensorDataset(train_tokens, train_targets, train_task_ids, train_categories)
    val_dataset = TensorDataset(val_tokens, val_targets, val_task_ids, val_categories)
    
    # Create standard data loaders
    standard_train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create stratified data loader
    stratified_sampler = StratifiedBatchSampler(train_categories.view(-1), batch_size=32)
    stratified_train_loader = DataLoader(train_dataset, batch_sampler=stratified_sampler)
    
    # Create test tensors for visualization
    test_tokens = tokens[:2]  # First 2 batches
    test_categories = categories[:2]
    
    return standard_train_loader, stratified_train_loader, val_loader, test_tokens, test_categories


def visualize_phases(model, tokens, categories, output_dir, label=""):
    """
    Visualize phase vectors and similarity matrices.
    
    Args:
        model: Model with thalamus and phase generator
        tokens: Input tensor [B, N, D]
        categories: Category IDs [B, N]
        output_dir: Directory to save visualizations
        label: Label for the visualization files
        
    Returns:
        metrics: Dictionary of phase similarity metrics
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
        
        # Extract phase vectors (assuming d_model is the token dimension)
        d_model = tokens.size(-1)
        phases = gated[..., d_model:]
    
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
    ax, metrics = plot_intra_inter_similarity(
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
    
    return metrics


def compare_training_configurations(output_dir):
    """
    Compare different contrastive learning configurations.
    
    Args:
        output_dir: Directory to save results
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
    num_epochs = 20
    
    # Phase generator configuration
    phase_config = {
        'hidden_dims': [256, 128, 64],
        'activation': 'gelu',
        'phase_diversity': 2.0,
        'use_layer_norm': True
    }
    
    # Create synthetic data
    print("Creating synthetic data...")
    standard_loader, stratified_loader, val_loader, test_tokens, test_categories = create_synthetic_data(
        num_samples=1000,
        num_tokens=32,
        d_model=d_model,
        num_categories=4,
        num_classes=num_classes
    )
    
    # Define training configurations to compare
    configs = [
        {
            'name': 'baseline',
            'desc': 'Basic contrastive learning',
            'params': {
                'contrastive_weight': 0.5,
                'temperature': 0.5,
                'use_temperature_scheduler': False,
                'use_adaptive_weighting': False,
                'use_hard_negatives': False,
                'monitor_gradients': False
            },
            'loader': standard_loader
        },
        {
            'name': 'temp_scheduling',
            'desc': 'With temperature scheduling',
            'params': {
                'contrastive_weight': 0.5,
                'temperature': 1.0,
                'use_temperature_scheduler': True,
                'use_adaptive_weighting': False,
                'use_hard_negatives': False,
                'monitor_gradients': False
            },
            'loader': standard_loader
        },
        {
            'name': 'hard_negatives',
            'desc': 'With hard negative mining',
            'params': {
                'contrastive_weight': 0.5,
                'temperature': 0.5,
                'use_temperature_scheduler': False,
                'use_adaptive_weighting': False,
                'use_hard_negatives': True,
                'hard_negative_ratio': 0.5,
                'monitor_gradients': False
            },
            'loader': standard_loader
        },
        {
            'name': 'adaptive_weighting',
            'desc': 'With adaptive loss weighting',
            'params': {
                'contrastive_weight': 0.5,
                'temperature': 0.5,
                'use_temperature_scheduler': False,
                'use_adaptive_weighting': True,
                'use_hard_negatives': False,
                'monitor_gradients': True
            },
            'loader': standard_loader
        },
        {
            'name': 'stratified_batching',
            'desc': 'With stratified batch sampling',
            'params': {
                'contrastive_weight': 0.5,
                'temperature': 0.5,
                'use_temperature_scheduler': False,
                'use_adaptive_weighting': False,
                'use_hard_negatives': False,
                'monitor_gradients': False
            },
            'loader': stratified_loader
        },
        {
            'name': 'full_enhanced',
            'desc': 'All enhancements combined',
            'params': {
                'contrastive_weight': 0.5,
                'temperature': 1.0,
                'use_temperature_scheduler': True,
                'use_adaptive_weighting': True,
                'use_hard_negatives': True,
                'hard_negative_ratio': 0.5,
                'monitor_gradients': True,
                'total_epochs': num_epochs
            },
            'loader': stratified_loader
        }
    ]
    
    # Train and evaluate each configuration
    all_metrics = {}
    before_metrics = {}
    after_metrics = {}
    
    for config in configs:
        print(f"\n\n{'='*50}")
        print(f"Training configuration: {config['name']} - {config['desc']}")
        print(f"{'='*50}")
        
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
        
        # Visualize phases before training
        config_dir = os.path.join(output_dir, config['name'])
        os.makedirs(config_dir, exist_ok=True)
        
        print(f"\nVisualizing phases before training...")
        before_metrics[config['name']] = visualize_phases(
            model=model,
            tokens=test_tokens,
            categories=test_categories,
            output_dir=config_dir,
            label="before_training"
        )
        
        # Create trainer
        trainer = ContrastivePhaseTrainer(
            model=model,
            **config['params']
        )
        
        # Train the model
        print(f"\nTraining with {config['desc']}...")
        train_metrics = trainer.train(
            train_loader=config['loader'],
            val_loader=val_loader,
            num_epochs=num_epochs,
            learning_rate=0.001,
            weight_decay=1e-5,
            task_id_key=2,  # Index of task_ids in batch
            category_key=3,  # Index of categories in batch
            save_dir=config_dir,
            save_freq=num_epochs  # Save only at the end
        )
        
        all_metrics[config['name']] = train_metrics
        
        # Visualize phases after training
        print(f"\nVisualizing phases after training...")
        after_metrics[config['name']] = visualize_phases(
            model=model,
            tokens=test_tokens,
            categories=test_categories,
            output_dir=config_dir,
            label="after_training"
        )
    
    # Compare results across configurations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Validation accuracy comparison
    plt.figure(figsize=(12, 6))
    for name, metrics in all_metrics.items():
        plt.plot(metrics['val_accuracy'], label=name)
    
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, f"accuracy_comparison_{timestamp}.png"))
    plt.close()
    
    # 2. Contrastive loss comparison
    plt.figure(figsize=(12, 6))
    for name, metrics in all_metrics.items():
        plt.plot(metrics['train_contrastive_loss'], label=name)
    
    plt.title('Contrastive Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, f"contrastive_loss_comparison_{timestamp}.png"))
    plt.close()
    
    # 3. Phase contrast comparison (before vs after)
    labels = list(configs)
    before_contrasts = [before_metrics[name]['contrast'] for name in labels]
    after_contrasts = [after_metrics[name]['contrast'] for name in labels]
    improvement = [after - before for after, before in zip(after_contrasts, before_contrasts)]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(labels))
    width = 0.25
    
    ax.bar(x - width, before_contrasts, width, label='Before Training')
    ax.bar(x, after_contrasts, width, label='After Training')
    ax.bar(x + width, improvement, width, label='Improvement')
    
    ax.set_title('Category Contrast Comparison (Before vs After Training)')
    ax.set_xlabel('Training Configuration')
    ax.set_ylabel('Contrast (Intra-Cat Sim - Inter-Cat Sim)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Add value labels
    for i, v in enumerate(before_contrasts):
        ax.text(i - width, v + 0.01, f"{v:.3f}", ha='center')
    for i, v in enumerate(after_contrasts):
        ax.text(i, v + 0.01, f"{v:.3f}", ha='center')
    for i, v in enumerate(improvement):
        ax.text(i + width, v + 0.01, f"{v:.3f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"contrast_comparison_{timestamp}.png"))
    plt.close()
    
    # Print summary of results
    print("\n\nResults Summary:")
    print("="*80)
    print(f"{'Configuration':<20} {'Before Contrast':<15} {'After Contrast':<15} {'Improvement':<15} {'Final Accuracy':<15}")
    print("-"*80)
    
    for name in labels:
        before = before_metrics[name]['contrast']
        after = after_metrics[name]['contrast']
        improv = after - before
        acc = all_metrics[name]['val_accuracy'][-1]
        
        print(f"{name:<20} {before:<15.4f} {after:<15.4f} {improv:<15.4f} {acc:<15.4f}")
    
    print("="*80)
    print(f"Results saved to {output_dir}")


def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Advanced Contrastive Learning Demo')
    parser.add_argument('--output_dir', type=str, default='advanced_contrastive_results',
                      help='Directory to save results')
    parser.add_argument('--no_comparison', action='store_true',
                      help='Skip configuration comparison (faster)')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    if args.no_comparison:
        # Train a single enhanced model
        print("Training single enhanced model...")
        
        # Create model and data
        d_model = 128
        phase_config = {
            'hidden_dims': [256, 128, 64],
            'activation': 'gelu',
            'phase_diversity': 2.0,
            'use_layer_norm': True
        }
        
        model = SimpleModel(
            d_model=d_model,
            n_heads=4,
            k=16,
            phase_dim=16,
            task_dim=64,
            num_tasks=10,
            hidden_dim=256,
            output_dim=10,
            phase_config=phase_config
        )
        
        # Create synthetic data
        _, train_loader, val_loader, test_tokens, test_categories = create_synthetic_data(
            num_samples=1000,
            num_tokens=32,
            d_model=d_model,
            num_categories=4,
            num_classes=10
        )
        
        # Visualize before training
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        before_metrics = visualize_phases(
            model=model,
            tokens=test_tokens,
            categories=test_categories,
            output_dir=output_dir,
            label="before_training"
        )
        
        # Create enhanced trainer
        trainer = ContrastivePhaseTrainer(
            model=model,
            contrastive_weight=0.5,
            temperature=1.0,
            use_temperature_scheduler=True,
            use_adaptive_weighting=True,
            use_hard_negatives=True,
            hard_negative_ratio=0.5,
            monitor_gradients=True,
            total_epochs=20
        )
        
        # Train the model
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=20,
            learning_rate=0.001,
            weight_decay=1e-5,
            task_id_key=2,
            category_key=3,
            save_dir=output_dir,
            save_freq=5
        )
        
        # Visualize after training
        after_metrics = visualize_phases(
            model=model,
            tokens=test_tokens,
            categories=test_categories,
            output_dir=output_dir,
            label="after_training"
        )
        
        # Print improvement
        print("\nCategory Contrast Metrics:")
        print(f"Before training: {before_metrics['contrast']:.4f}")
        print(f"After training:  {after_metrics['contrast']:.4f}")
        print(f"Improvement:     {after_metrics['contrast'] - before_metrics['contrast']:.4f}")
        
    else:
        # Compare different configurations
        compare_training_configurations(args.output_dir)


if __name__ == "__main__":
    main()
