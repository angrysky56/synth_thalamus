"""
Advanced Contrastive Learning Utilities for Synthetic Thalamus

This module provides specialized utilities for contrastive learning, including:
- Enhanced contrastive loss with hard negative mining
- Temperature scheduling
- Adaptive loss weighting
- Gradient monitoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from torch.utils.data import Sampler


class TemperatureScheduler:
    """
    Scheduler for the temperature parameter in contrastive loss.
    
    This can implement various schedules:
    - Linear: Linearly interpolate between start and end temperatures
    - Cosine: Use cosine annealing schedule
    - Step: Decrease temperature at specified steps
    """
    def __init__(self, schedule_type='cosine', start_temp=1.0, end_temp=0.1, 
                 total_epochs=100, warmup_epochs=10):
        """
        Initialize the temperature scheduler.
        
        Args:
            schedule_type: Type of schedule ('linear', 'cosine', 'step')
            start_temp: Initial temperature
            end_temp: Final temperature
            total_epochs: Total number of training epochs
            warmup_epochs: Number of warmup epochs
        """
        self.schedule_type = schedule_type
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        
    def get_temperature(self, epoch):
        """
        Get temperature for the current epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            temperature: Temperature value for the current epoch
        """
        if epoch < self.warmup_epochs:
            # Warmup phase: linearly increase from end_temp to start_temp
            alpha = epoch / self.warmup_epochs
            return self.end_temp + alpha * (self.start_temp - self.end_temp)
        
        if self.schedule_type == 'linear':
            # Linear schedule
            progress = min(1.0, (epoch - self.warmup_epochs) / 
                         (self.total_epochs - self.warmup_epochs))
            return self.start_temp + progress * (self.end_temp - self.start_temp)
        
        elif self.schedule_type == 'cosine':
            # Cosine annealing schedule
            progress = min(1.0, (epoch - self.warmup_epochs) / 
                         (self.total_epochs - self.warmup_epochs))
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
            return self.end_temp + cosine_factor * (self.start_temp - self.end_temp)
        
        elif self.schedule_type == 'step':
            # Step schedule (decrease by half every 1/3 of training)
            step_size = (self.total_epochs - self.warmup_epochs) / 3
            step = min(3, int((epoch - self.warmup_epochs) / step_size) + 1)
            return self.start_temp * (0.5 ** (step - 1))
        
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")


class AdaptiveLossWeighter:
    """
    Adaptively weight task and contrastive losses based on training progress
    and gradient statistics.
    """
    def __init__(self, init_weight=0.5, target_weight=1.0, total_epochs=100, 
                 schedule_type='linear', ema_decay=0.9):
        """
        Initialize the adaptive loss weighter.
        
        Args:
            init_weight: Initial weight for contrastive loss
            target_weight: Target weight for contrastive loss
            total_epochs: Total number of training epochs
            schedule_type: Type of schedule ('linear', 'cosine')
            ema_decay: Decay factor for exponential moving average
        """
        self.init_weight = init_weight
        self.target_weight = target_weight
        self.total_epochs = total_epochs
        self.schedule_type = schedule_type
        self.ema_decay = ema_decay
        
        # For tracking gradient statistics
        self.task_grad_norm_ema = None
        self.contrastive_grad_norm_ema = None
        
    def get_contrastive_weight(self, epoch, task_grad_norm=None, contrastive_grad_norm=None):
        """
        Get weight for contrastive loss based on scheduling and gradient statistics.
        
        Args:
            epoch: Current epoch number
            task_grad_norm: Norm of task loss gradients (optional)
            contrastive_grad_norm: Norm of contrastive loss gradients (optional)
            
        Returns:
            weight: Weight for contrastive loss
        """
        # Update EMA of gradient norms if provided
        if task_grad_norm is not None and contrastive_grad_norm is not None:
            if self.task_grad_norm_ema is None:
                self.task_grad_norm_ema = task_grad_norm
                self.contrastive_grad_norm_ema = contrastive_grad_norm
            else:
                self.task_grad_norm_ema = self.ema_decay * self.task_grad_norm_ema + \
                                         (1 - self.ema_decay) * task_grad_norm
                self.contrastive_grad_norm_ema = self.ema_decay * self.contrastive_grad_norm_ema + \
                                               (1 - self.ema_decay) * contrastive_grad_norm
        
        # Get scheduled weight based on epoch
        if self.schedule_type == 'linear':
            progress = min(1.0, epoch / self.total_epochs)
            scheduled_weight = self.init_weight + progress * (self.target_weight - self.init_weight)
        elif self.schedule_type == 'cosine':
            progress = min(1.0, epoch / self.total_epochs)
            cosine_factor = 0.5 * (1 + np.cos(np.pi * (1 - progress)))
            scheduled_weight = self.init_weight + cosine_factor * (self.target_weight - self.init_weight)
        else:
            scheduled_weight = self.init_weight
            
        # Adjust weight based on gradient statistics if available
        if self.task_grad_norm_ema is not None and self.contrastive_grad_norm_ema is not None:
            # Balance the contributions by scaling the contrastive weight
            # inversely proportional to the ratio of gradient norms
            grad_ratio = self.contrastive_grad_norm_ema / (self.task_grad_norm_ema + 1e-8)
            
            # Limit the adjustment factor to avoid extreme values
            adjustment_factor = min(max(1.0 / grad_ratio, 0.1), 10.0)
            
            # Apply a softer adjustment by taking the square root
            soft_adjustment = np.sqrt(adjustment_factor)
            
            return scheduled_weight * soft_adjustment
        else:
            return scheduled_weight


class GradientMonitor:
    """
    Monitor gradient statistics for different components of the model.
    """
    def __init__(self, model, log_freq=10):
        """
        Initialize the gradient monitor.
        
        Args:
            model: Model to monitor
            log_freq: Frequency of logging (in steps)
        """
        self.model = model
        self.log_freq = log_freq
        self.step = 0
        self.stats = defaultdict(list)
        
        # Register hooks for the phase generator components
        self.hooks = []
        for name, param in model.named_parameters():
            if 'phase_generator' in name:
                hook = param.register_hook(
                    lambda grad, name=name: self._hook_fn(grad, name))
                self.hooks.append(hook)
    
    def _hook_fn(self, grad, name):
        """
        Hook function to capture gradient statistics.
        
        Args:
            grad: Gradient tensor
            name: Parameter name
        """
        if self.step % self.log_freq == 0:
            norm = grad.norm().item()
            self.stats[f"{name}_norm"].append(norm)
            
            # Additional statistics
            if grad.numel() > 1:
                mean = grad.mean().item()
                std = grad.std().item()
                self.stats[f"{name}_mean"].append(mean)
                self.stats[f"{name}_std"].append(std)
                
                # Check for potential gradient issues
                if norm < 1e-4:
                    self.stats[f"{name}_vanishing"].append(self.step)
                if norm > 10.0:
                    self.stats[f"{name}_exploding"].append(self.step)
        
        # Return gradient unchanged
        return grad
    
    def step_callback(self):
        """Call this after each optimization step."""
        self.step += 1
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def plot_gradient_stats(self, figsize=(12, 8)):
        """
        Plot gradient statistics.
        
        Args:
            figsize: Figure size
            
        Returns:
            fig: Matplotlib figure
        """
        layers = set()
        for key in self.stats.keys():
            if '_norm' in key:
                layer = key.replace('_norm', '')
                layers.add(layer)
        
        fig, axes = plt.subplots(len(layers), 1, figsize=figsize)
        if len(layers) == 1:
            axes = [axes]
        
        for i, layer in enumerate(sorted(layers)):
            ax = axes[i]
            steps = list(range(0, self.step, self.log_freq))[:len(self.stats[f"{layer}_norm"])]
            
            ax.plot(steps, self.stats[f"{layer}_norm"], label='Gradient Norm')
            if f"{layer}_mean" in self.stats:
                ax.plot(steps, self.stats[f"{layer}_mean"], label='Gradient Mean', alpha=0.7)
            if f"{layer}_std" in self.stats:
                ax.plot(steps, self.stats[f"{layer}_std"], label='Gradient Std', alpha=0.7)
            
            ax.set_title(f'Gradient Stats: {layer}')
            ax.set_xlabel('Step')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


class StratifiedBatchSampler(Sampler):
    """
    Stratified batch sampler to ensure balanced category representation in each batch.
    """
    def __init__(self, categories, batch_size, drop_last=False):
        """
        Initialize the stratified batch sampler.
        
        Args:
            categories: Tensor of category IDs for each sample
            batch_size: Batch size
            drop_last: Whether to drop the last incomplete batch
        """
        self.categories = categories
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Determine total dataset size (number of actual samples)
        if isinstance(categories, torch.Tensor) and categories.dim() > 1:
            # If categories is a 2D tensor, we're dealing with a batched dataset
            # Each row is a separate sample
            self.total_samples = categories.size(0)
            # Flatten categories for better indexing
            categories_flat = categories.view(-1)
            # Create index mapping from flat to original indices
            self.index_map = {i: (i // categories.size(1), i % categories.size(1)) 
                             for i in range(len(categories_flat))}
            cat_tensor = categories_flat
        else:
            # Otherwise, assume 1D tensor where each element is a category
            self.total_samples = len(categories)
            self.index_map = None
            cat_tensor = categories
            
        # Group indices by category
        self.category_indices = defaultdict(list)
        for idx, cat in enumerate(cat_tensor):
            # For flattened dataset, only include indices that map to actual samples
            if self.index_map is None or idx // categories.size(1) < self.total_samples:
                self.category_indices[cat.item()].append(idx)
            
        # Determine number of samples per category in each batch
        self.unique_categories = list(self.category_indices.keys())
        num_cats = max(1, len(self.unique_categories))  # Avoid division by zero
        self.samples_per_cat = max(1, batch_size // num_cats)  # At least 1 sample per category
        
    def __iter__(self):
        """
        Iterate over batches.
        
        Yields:
            batch: List of indices for the current batch
        """
        # Safety check to prevent infinite loops
        if self.total_samples == 0 or not self.unique_categories:
            return
            
        # Shuffle indices within each category
        for cat in self.unique_categories:
            random.shuffle(self.category_indices[cat])
            
        # Create iterators for each category
        category_iterators = {}
        for cat in self.unique_categories:
            # Only create iterators for categories with samples
            if self.category_indices[cat]:
                category_iterators[cat] = iter(self.category_indices[cat])
        
        # Keep track of which categories are exhausted
        exhausted_categories = set()
        
        # Generate batches until we've used all samples from all categories
        while len(exhausted_categories) < len(self.unique_categories):
            batch = []
            
            # Try to sample from each category
            for cat in self.unique_categories:
                if cat in exhausted_categories:
                    continue
                    
                try:
                    # Add samples from this category up to samples_per_cat
                    samples_to_add = min(self.samples_per_cat, self.batch_size - len(batch))
                    for _ in range(samples_to_add):
                        if len(batch) >= self.batch_size:
                            break
                        batch.append(next(category_iterators[cat]))
                except (StopIteration, KeyError):
                    # This category is exhausted
                    exhausted_categories.add(cat)
            
            # If we couldn't add any samples, we're done
            if not batch:
                break
                
            # If we need to fill the batch and not all categories are exhausted
            remaining = self.batch_size - len(batch)
            if remaining > 0 and len(exhausted_categories) < len(self.unique_categories):
                # Use available categories to fill the batch
                available_cats = [cat for cat in self.unique_categories 
                                if cat not in exhausted_categories]
                
                while remaining > 0 and available_cats:
                    cat = random.choice(available_cats)
                    try:
                        batch.append(next(category_iterators[cat]))
                        remaining -= 1
                    except StopIteration:
                        # This category is now exhausted
                        exhausted_categories.add(cat)
                        available_cats.remove(cat)
            
            # Only yield batch if it's complete or we don't need to drop incomplete batches
            if len(batch) == self.batch_size or (not self.drop_last and batch):
                # Shuffle the batch to avoid category-based ordering
                random.shuffle(batch)
                
                # If we're dealing with a 2D tensor, map flat indices back to sample indices
                if self.index_map is not None:
                    # Extract only the sample index from the mapping
                    batch = [self.index_map[idx][0] for idx in batch]
                    
                    # Remove any duplicates (could happen with the flattened indexing)
                    batch = list(set(batch))
                    
                    # If after removing duplicates the batch is too small, skip it
                    if len(batch) < 1:
                        continue
                
                yield batch
                
    def __len__(self):
        """Return the number of batches."""
        # Estimate the number of batches conservatively
        if self.drop_last:
            return self.total_samples // self.batch_size
        else:
            return (self.total_samples + self.batch_size - 1) // self.batch_size


def contrastive_loss_with_hard_negatives(phase_vectors, semantic_categories, temperature=0.5, hard_negative_ratio=0.5):
    """
    Contrastive loss with focus on hard negatives.
    
    Args:
        phase_vectors: Tensor of shape [B, k, phase_dim]
        semantic_categories: Tensor of shape [B, k] containing category IDs
        temperature: Temperature parameter for scaling similarities
        hard_negative_ratio: Ratio of hard negatives to use (0-1)
        
    Returns:
        loss: Contrastive loss value
    """
    # Print debug information
    print(f"Phase vectors shape: {phase_vectors.shape}")
    print(f"Semantic categories shape: {semantic_categories.shape}")
    
    # Ensure semantic_categories has the right shape
    if semantic_categories.dim() > phase_vectors.dim() - 1:
        # Too many dimensions - need to reduce
        if semantic_categories.dim() == 3 and phase_vectors.dim() == 3:
            # Both are batched but categories might have extra dimension
            if semantic_categories.size(2) > 1:
                # Take the first category dimension if there are multiple
                semantic_categories = semantic_categories[:, :, 0]
            else:
                # Squeeze out the extra dimension
                semantic_categories = semantic_categories.squeeze(-1)
        elif semantic_categories.dim() == 2 and phase_vectors.dim() == 2:
            # Unbatched case
            semantic_categories = semantic_categories[:, 0]
        else:
            # Use reshape to match dimensions in other cases
            semantic_categories = semantic_categories.reshape(phase_vectors.size(0), phase_vectors.size(1))
    elif semantic_categories.dim() < phase_vectors.dim() - 1:
        # Not enough dimensions - need to add
        if semantic_categories.dim() == 1 and phase_vectors.dim() == 3:
            # Add batch dimension
            semantic_categories = semantic_categories.unsqueeze(0)
    
    # Final check for the correct shape
    if phase_vectors.dim() == 3 and semantic_categories.dim() == 2:
        if phase_vectors.size(1) != semantic_categories.size(1):
            print(f"Warning: Dimension mismatch. Phase vectors length {phase_vectors.size(1)}, "
                  f"Categories length {semantic_categories.size(1)}")
            
            # Reduce both to the minimum size
            min_size = min(phase_vectors.size(1), semantic_categories.size(1))
            phase_vectors = phase_vectors[:, :min_size]
            semantic_categories = semantic_categories[:, :min_size]
            
            print(f"Adjusted shapes: Phase vectors {phase_vectors.shape}, "
                  f"Categories {semantic_categories.shape}")
    
    # Print final shapes
    print(f"Final shapes: Phase vectors {phase_vectors.shape}, Categories {semantic_categories.shape}")
    
    # Normalize phase vectors
    phase_norm = F.normalize(phase_vectors, p=2, dim=2)
    
    # Calculate pairwise cosine similarity
    similarity = torch.bmm(phase_norm, phase_norm.transpose(1, 2))
    
    # Create mask of positive pairs (same semantic category)
    pos_mask = (semantic_categories.unsqueeze(2) == semantic_categories.unsqueeze(1))
    
    # Create mask for valid comparisons (exclude self-comparisons)
    diag_mask = ~torch.eye(pos_mask.size(1), dtype=torch.bool, 
                         device=pos_mask.device).unsqueeze(0)
    
    # Debugging info
    print(f"Similarity shape: {similarity.shape}")
    print(f"Positive mask shape: {pos_mask.shape}")
    
    # Create negative mask (different semantic category)
    neg_mask = ~pos_mask & diag_mask
    
    # For each anchor, find the hardest negatives (highest similarity)
    neg_sim = similarity * neg_mask.float()
    
    # For each row, get indices sorted by similarity (descending)
    _, hard_indices = torch.sort(neg_sim, dim=2, descending=True)
    
    # Determine how many hard negatives to use per anchor
    num_negatives = neg_mask.sum(dim=2)
    num_hard_negatives = (num_negatives * hard_negative_ratio).long()
    
    # Create hard negative mask
    hard_neg_mask = torch.zeros_like(neg_mask)
    for b in range(hard_neg_mask.size(0)):
        for i in range(hard_neg_mask.size(1)):
            if num_hard_negatives[b, i] > 0:
                # Get indices of hard negatives for this anchor
                indices = hard_indices[b, i, :num_hard_negatives[b, i]]
                hard_neg_mask[b, i, indices] = True
    
    # Apply temperature scaling
    similarity = similarity / temperature
    
    # InfoNCE-style contrastive loss with hard negatives
    exp_sim = torch.exp(similarity)
    
    # For each token, sum similarities with positives
    pos_sim = torch.sum(exp_sim * pos_mask * diag_mask, dim=2)
    
    # Sum of similarities with hard negatives
    hard_neg_sim = torch.sum(exp_sim * hard_neg_mask, dim=2)
    
    # Use all positives but only hard negatives for the denominator
    # Add small epsilon for numerical stability
    eps = 1e-8
    loss = -torch.log((pos_sim + eps) / (pos_sim + hard_neg_sim + eps))
    
    # Average over all tokens
    return loss.mean()
