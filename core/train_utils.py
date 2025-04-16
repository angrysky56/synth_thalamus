"""
Training utilities for the Synthetic Thalamus

This module provides tools for training models with the Synthetic Thalamus,
including contrastive learning for the phase generator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import time
from tqdm import tqdm
from collections import defaultdict

from core.phase_generator import contrastive_loss


from .contrastive_utils import (
    TemperatureScheduler, AdaptiveLossWeighter, GradientMonitor,
    contrastive_loss_with_hard_negatives
)
from .phase_generator import contrastive_loss

class ContrastivePhaseTrainer:
    """
    Trainer for contrastive learning with the enhanced phase generator.
    
    This class provides utilities for training a model with the Synthetic Thalamus,
    incorporating contrastive learning to improve phase vector quality.
    """
    
    def __init__(self, model, device=None, 
                 contrastive_weight=0.5, 
                 temperature=0.5,
                 use_temperature_scheduler=False,
                 use_adaptive_weighting=False,
                 use_hard_negatives=False,
                 hard_negative_ratio=0.5,
                 monitor_gradients=False,
                 total_epochs=100):
        """
        Initialize the contrastive phase trainer.
        
        Args:
            model: Model containing a thalamus with enhanced phase generator
            device: Device to run the model on (cpu or cuda)
            contrastive_weight: Initial weight of the contrastive loss term
            temperature: Initial temperature parameter for contrastive loss
            use_temperature_scheduler: Whether to use temperature scheduling
            use_adaptive_weighting: Whether to use adaptive loss weighting
            use_hard_negatives: Whether to use hard negative mining
            hard_negative_ratio: Ratio of hard negatives to use (0-1)
            monitor_gradients: Whether to monitor gradient statistics
            total_epochs: Total number of epochs for scheduling
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Basic contrastive learning parameters
        self.initial_contrastive_weight = contrastive_weight
        self.contrastive_weight = contrastive_weight
        self.initial_temperature = temperature
        self.temperature = temperature
        
        # Advanced contrastive learning features
        self.use_temperature_scheduler = use_temperature_scheduler
        self.use_adaptive_weighting = use_adaptive_weighting
        self.use_hard_negatives = use_hard_negatives
        self.hard_negative_ratio = hard_negative_ratio
        self.monitor_gradients = monitor_gradients
        self.total_epochs = total_epochs
        
        # Initialize schedulers if enabled
        if use_temperature_scheduler:
            self.temp_scheduler = TemperatureScheduler(
                schedule_type='cosine',
                start_temp=temperature,
                end_temp=0.1,
                total_epochs=total_epochs
            )
            
        if use_adaptive_weighting:
            self.loss_weighter = AdaptiveLossWeighter(
                init_weight=contrastive_weight,
                target_weight=1.0,
                total_epochs=total_epochs,
                schedule_type='cosine'
            )
            
        # Initialize gradient monitor if enabled
        if monitor_gradients:
            self.grad_monitor = GradientMonitor(model)
            
        # Track metrics
        self.metrics = defaultdict(list)
        self.current_epoch = 0
    
    def contrastive_loss_fn(self, phase_vectors, semantic_categories):
        """
        Calculate contrastive loss with current settings.
        
        Args:
            phase_vectors: Tensor of shape [B, k, phase_dim]
            semantic_categories: Tensor of shape [B, k] containing category IDs
            
        Returns:
            loss: Contrastive loss value
        """
        # Get current temperature (scheduled or fixed)
        if self.use_temperature_scheduler:
            temperature = self.temp_scheduler.get_temperature(self.current_epoch)
        else:
            temperature = self.temperature
            
        # Use standard or hard negative contrastive loss
        if self.use_hard_negatives:
            return contrastive_loss_with_hard_negatives(
                phase_vectors, 
                semantic_categories, 
                temperature=temperature,
                hard_negative_ratio=self.hard_negative_ratio
            )
        else:
            return contrastive_loss(
                phase_vectors, 
                semantic_categories, 
                temperature=temperature
            )
            
    def train_epoch(self, train_loader, optimizer, task_id_key=None, category_key=None):
        """
        Train for one epoch with contrastive learning.
        
        Args:
            train_loader: DataLoader providing training batches
            optimizer: PyTorch optimizer
            task_id_key: Key to extract task IDs from batch dictionary
            category_key: Key to extract semantic categories from batch dictionary
            
        Returns:
            metrics: Dictionary of training metrics
        """
        self.model.train()
        epoch_loss = 0
        epoch_task_loss = 0
        epoch_contrastive_loss = 0
        
        # Update temperature and contrastive weight based on schedulers
        if self.use_temperature_scheduler:
            self.temperature = self.temp_scheduler.get_temperature(self.current_epoch)
            
        if self.use_adaptive_weighting:
            self.contrastive_weight = self.loss_weighter.get_contrastive_weight(self.current_epoch)
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {self.current_epoch+1} Training")):
            # Extract data from batch
            if isinstance(batch, dict):
                x = batch['x'].to(self.device)
                y = batch['y'].to(self.device) if 'y' in batch else None
                
                task_ids = batch[task_id_key].to(self.device) if task_id_key and task_id_key in batch else None
                categories = batch[category_key].to(self.device) if category_key and category_key in batch else None
            else:
                # Assume batch is a tuple or list (x, y, ...)
                x = batch[0].to(self.device)
                y = batch[1].to(self.device) if len(batch) > 1 else None
                
                task_ids = batch[2].to(self.device) if len(batch) > 2 and task_id_key is not None else None
                categories = batch[3].to(self.device) if len(batch) > 3 and category_key is not None else None
            
            optimizer.zero_grad()
            
            # Setup for capturing gradients and phases
            task_grads = []
            contrastive_grads = []
            phases = None
            
            # Define hook to capture task loss gradients if needed
            if self.use_adaptive_weighting and self.monitor_gradients:
                def task_backward_hook(grad):
                    task_grads.append(grad.detach().clone())
                    return grad
                
                def contrastive_backward_hook(grad):
                    contrastive_grads.append(grad.detach().clone())
                    return grad
            
            # Capture phase vectors
            with torch.no_grad():
                # Register forward hook to capture phases
                phase_outputs = {}
                
                def forward_hook(module, input, output):
                    phase_outputs['phases'] = output
                
                # Add hook to the phase generator
                if hasattr(self.model, 'thalamus') and hasattr(self.model.thalamus, 'phase_generator'):
                    hook_handle = self.model.thalamus.phase_generator.register_forward_hook(forward_hook)
                    
                    # Forward pass to get phases
                    _ = self.model(x, task_ids) if task_ids is not None else self.model(x)
                    
                    phases = phase_outputs.get('phases')
                    hook_handle.remove()
            
            # Forward pass for real
            outputs = self.model(x, task_ids) if task_ids is not None else self.model(x)
            
            # Compute task loss
            task_loss = F.cross_entropy(outputs, y) if y is not None else torch.tensor(0.0, device=self.device)
            
            # Compute contrastive loss if possible
            c_loss = torch.tensor(0.0, device=self.device)
            
            if phases is not None and categories is not None:
                phases.requires_grad_(True)
                
                # Register hooks for gradient analysis if needed
                if self.use_adaptive_weighting and self.monitor_gradients:
                    task_hook = phases.register_hook(task_backward_hook)
                
                # Forward pass for task loss with phases
                task_loss.backward(retain_graph=True)
                
                if self.use_adaptive_weighting and self.monitor_gradients:
                    task_hook.remove()
                
                # Calculate contrastive loss
                c_loss = self.contrastive_loss_fn(phases, categories)
                
                # Register hook for contrastive gradients if needed
                if self.use_adaptive_weighting and self.monitor_gradients:
                    contrastive_hook = phases.register_hook(contrastive_backward_hook)
                
                # Backward pass for contrastive loss
                (self.contrastive_weight * c_loss).backward()
                
                if self.use_adaptive_weighting and self.monitor_gradients:
                    contrastive_hook.remove()
                    
                    # Calculate gradient norms for loss balancing
                    if task_grads and contrastive_grads:
                        task_grad_norm = task_grads[0].norm().item()
                        contrastive_grad_norm = contrastive_grads[0].norm().item()
                        
                        # Update loss weighter with gradient statistics
                        self.contrastive_weight = self.loss_weighter.get_contrastive_weight(
                            self.current_epoch, 
                            task_grad_norm, 
                            contrastive_grad_norm
                        )
            else:
                # If no phases or categories, just do task loss backward
                task_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            
            # Optimization step
            optimizer.step()
            
            if self.monitor_gradients:
                self.grad_monitor.step_callback()
            
            # Update metrics
            total_loss = task_loss.item() + self.contrastive_weight * c_loss.item()
            epoch_loss += total_loss
            epoch_task_loss += task_loss.item()
            epoch_contrastive_loss += c_loss.item()
        
        # Compute average metrics
        num_batches = len(train_loader)
        avg_loss = epoch_loss / num_batches
        avg_task_loss = epoch_task_loss / num_batches
        avg_contrastive_loss = epoch_contrastive_loss / num_batches
        
        # Store metrics
        self.metrics['train_loss'].append(avg_loss)
        self.metrics['train_task_loss'].append(avg_task_loss)
        self.metrics['train_contrastive_loss'].append(avg_contrastive_loss)
        self.metrics['contrastive_weight'].append(self.contrastive_weight)
        self.metrics['temperature'].append(self.temperature)
        
        # Increment epoch counter
        self.current_epoch += 1
        
        return {
            'loss': avg_loss,
            'task_loss': avg_task_loss,
            'contrastive_loss': avg_contrastive_loss,
            'contrastive_weight': self.contrastive_weight,
            'temperature': self.temperature
        }
    
    def validate(self, val_loader, task_id_key=None):
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader providing validation batches
            task_id_key: Key to extract task IDs from batch dictionary
            
        Returns:
            metrics: Dictionary of validation metrics
        """
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
                # Extract data from batch
                if isinstance(batch, dict):
                    x = batch['x'].to(self.device)
                    y = batch['y'].to(self.device) if 'y' in batch else None
                    
                    task_ids = batch[task_id_key].to(self.device) if task_id_key and task_id_key in batch else None
                else:
                    # Assume batch is a tuple or list (x, y, ...)
                    x = batch[0].to(self.device)
                    y = batch[1].to(self.device) if len(batch) > 1 else None
                    
                    task_ids = batch[2].to(self.device) if len(batch) > 2 and task_id_key is not None else None
                
                # Forward pass
                outputs = self.model(x, task_ids) if task_ids is not None else self.model(x)
                
                # Compute loss
                loss = F.cross_entropy(outputs, y) if y is not None else 0
                
                # Update metrics
                val_loss += loss.item()
                
                if y is not None:
                    _, predicted = torch.max(outputs.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
        
        # Compute average metrics
        num_batches = len(val_loader)
        avg_loss = val_loss / num_batches
        accuracy = correct / total if total > 0 else 0
        
        # Store metrics
        self.metrics['val_loss'].append(avg_loss)
        self.metrics['val_accuracy'].append(accuracy)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def train(self, train_loader, val_loader, num_epochs, learning_rate=0.001, 
             weight_decay=0, task_id_key=None, category_key=None,
             save_dir=None, save_freq=None):
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: DataLoader providing training batches
            val_loader: DataLoader providing validation batches
            num_epochs: Number of epochs to train
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            task_id_key: Key to extract task IDs from batch dictionary
            category_key: Key to extract semantic categories from batch dictionary
            save_dir: Directory to save model checkpoints
            save_freq: Frequency (in epochs) to save checkpoints
            
        Returns:
            metrics: Dictionary of training and validation metrics
        """
        # Update total epochs for schedulers if needed
        if self.use_temperature_scheduler:
            self.temp_scheduler.total_epochs = num_epochs
            
        if self.use_adaptive_weighting:
            self.loss_weighter.total_epochs = num_epochs
            
        # Create optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Create save directory if needed
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Reset epoch counter
        self.current_epoch = 0
        
        # Training loop
        start_time = time.time()
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(
                train_loader, optimizer, task_id_key=task_id_key, category_key=category_key)
            
            # Validate
            val_metrics = self.validate(val_loader, task_id_key=task_id_key)
            
            # Print metrics with enhanced information
            print(f"Train loss: {train_metrics['loss']:.4f}, Task loss: {train_metrics['task_loss']:.4f}, "
                  f"Contrastive loss: {train_metrics['contrastive_loss']:.4f}")
            print(f"Contrastive weight: {train_metrics['contrastive_weight']:.4f}, "
                  f"Temperature: {train_metrics['temperature']:.4f}")
            print(f"Val loss: {val_metrics['loss']:.4f}, Val accuracy: {val_metrics['accuracy']:.4f}")
            
            # Save checkpoint
            if save_dir and save_freq and (epoch + 1) % save_freq == 0:
                save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'contrastive_weight': train_metrics['contrastive_weight'],
                    'temperature': train_metrics['temperature'],
                }, save_path)
                print(f"Model saved to {save_path}")
                
                # Save gradient statistics if monitoring
                if self.monitor_gradients:
                    fig = self.grad_monitor.plot_gradient_stats()
                    grad_path = os.path.join(save_dir, f"grad_stats_epoch_{epoch+1}.png")
                    fig.savefig(grad_path)
                    plt.close(fig)
                    print(f"Gradient statistics saved to {grad_path}")
        
        end_time = time.time()
        print(f"\nTraining completed in {(end_time - start_time) / 60:.2f} minutes")
        
        # Clean up resources
        if self.monitor_gradients:
            self.grad_monitor.remove_hooks()
        
        return self.metrics
    
    def save_model(self, filepath):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.to(self.device)
        print(f"Model loaded from {filepath}")
