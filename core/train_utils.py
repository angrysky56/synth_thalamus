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


class ContrastivePhaseTrainer:
    """
    Trainer for contrastive learning with the enhanced phase generator.
    
    This class provides utilities for training a model with the Synthetic Thalamus,
    incorporating contrastive learning to improve phase vector quality.
    """
    
    def __init__(self, model, device=None, contrastive_weight=0.1, temperature=0.5):
        """
        Initialize the contrastive phase trainer.
        
        Args:
            model: Model containing a thalamus with enhanced phase generator
            device: Device to run the model on (cpu or cuda)
            contrastive_weight: Weight of the contrastive loss term
            temperature: Temperature parameter for contrastive loss
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        
        self.metrics = defaultdict(list)
    
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
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
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
            
            # Run forward pass, retrieving phase vectors if available
            phases = None
            if hasattr(self.model, 'thalamus') and hasattr(self.model.thalamus, 'phase_generator'):
                # Store original forward method
                original_forward = self.model.thalamus.forward
                
                # Define wrapper to capture phase vectors
                def forward_wrapper(*args, **kwargs):
                    gated = original_forward(*args, **kwargs)
                    
                    # Store the last generated phases for contrastive loss
                    nonlocal phases
                    phases = self.model.thalamus.phase_generator.mlp.last_output
                    
                    return gated
                
                # Add attribute to store last output
                if not hasattr(self.model.thalamus.phase_generator.mlp, 'last_output'):
                    # Modify the MLP to store its last output
                    original_mlp_forward = self.model.thalamus.phase_generator.mlp.forward
                    
                    def mlp_forward_wrapper(self_mlp, x):
                        output = original_mlp_forward(x)
                        self_mlp.last_output = output
                        return output
                    
                    self.model.thalamus.phase_generator.mlp.forward = lambda x: mlp_forward_wrapper(
                        self.model.thalamus.phase_generator.mlp, x)
                
                # Replace forward method temporarily
                self.model.thalamus.forward = forward_wrapper
            
            # Forward pass
            outputs = self.model(x, task_ids) if task_ids is not None else self.model(x)
            
            # Restore original forward method if it was replaced
            if hasattr(self.model, 'thalamus') and hasattr(self.model.thalamus, 'phase_generator'):
                self.model.thalamus.forward = original_forward
            
            # Compute task loss
            task_loss = F.cross_entropy(outputs, y) if y is not None else 0
            
            # Compute contrastive loss if possible
            c_loss = 0
            if phases is not None and categories is not None:
                c_loss = contrastive_loss(phases, categories, temperature=self.temperature)
            
            # Combined loss
            loss = task_loss + self.contrastive_weight * c_loss
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            epoch_task_loss += task_loss.item() if isinstance(task_loss, torch.Tensor) else task_loss
            epoch_contrastive_loss += c_loss.item() if isinstance(c_loss, torch.Tensor) else c_loss
        
        # Compute average metrics
        num_batches = len(train_loader)
        avg_loss = epoch_loss / num_batches
        avg_task_loss = epoch_task_loss / num_batches
        avg_contrastive_loss = epoch_contrastive_loss / num_batches
        
        # Store metrics
        self.metrics['train_loss'].append(avg_loss)
        self.metrics['train_task_loss'].append(avg_task_loss)
        self.metrics['train_contrastive_loss'].append(avg_contrastive_loss)
        
        return {
            'loss': avg_loss,
            'task_loss': avg_task_loss,
            'contrastive_loss': avg_contrastive_loss
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
        # Create optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Create save directory if needed
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Training loop
        start_time = time.time()
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(
                train_loader, optimizer, task_id_key=task_id_key, category_key=category_key)
            
            # Validate
            val_metrics = self.validate(val_loader, task_id_key=task_id_key)
            
            # Print metrics
            print(f"Train loss: {train_metrics['loss']:.4f}, Task loss: {train_metrics['task_loss']:.4f}, "
                  f"Contrastive loss: {train_metrics['contrastive_loss']:.4f}")
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
                }, save_path)
                print(f"Model saved to {save_path}")
        
        end_time = time.time()
        print(f"\nTraining completed in {(end_time - start_time) / 60:.2f} minutes")
        
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
