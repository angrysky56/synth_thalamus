"""
Enhanced Semantic Phase Generator Module

This module provides implementations for generating phase vectors from semantic embeddings,
with advanced features such as configurable MLP architectures, contrastive learning,
and phase diversity control.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedSemanticPhaseGenerator(nn.Module):
    """
    Enhanced MLP for mapping semantic embeddings to phase vectors.
    
    Features:
    - Configurable MLP architecture with variable depth and width
    - Multiple activation function options (GELU, Leaky ReLU, SiLU, PReLU)
    - Layer normalization for better gradient flow
    - Phase diversity scaling parameter
    """
    def __init__(
        self, 
        d_model, 
        phase_dim, 
        hidden_dims=[128, 64], 
        activation='gelu', 
        phase_diversity=2.0,
        use_layer_norm=True
    ):
        """
        Initialize the enhanced phase generator.
        
        Args:
            d_model: Dimension of input semantic embeddings
            phase_dim: Dimension of output phase vectors
            hidden_dims: List of hidden layer dimensions
            activation: Activation function type ('gelu', 'leaky_relu', 'silu', 'prelu', 'relu')
            phase_diversity: Scaling factor for output diversity
            use_layer_norm: Whether to use layer normalization between layers
        """
        super().__init__()
        
        # Build MLP layers
        layers = []
        input_dim = d_model
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(input_dim, hidden_dim))
            
            # Add layer normalization if specified
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            # Add activation based on type
            if activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            elif activation == 'silu':
                layers.append(nn.SiLU())
            elif activation == 'prelu':
                layers.append(nn.PReLU())
            else:
                layers.append(nn.ReLU())
                
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, phase_dim))
        
        # Create sequential model
        self.mlp = nn.Sequential(*layers)
        
        # Make phase_diversity a learnable parameter
        self.phase_diversity = nn.Parameter(torch.tensor(phase_diversity))
        
    def forward(self, x):
        """
        Generate phase vectors from semantic embeddings.
        
        Args:
            x: Tensor [B, k, d_model] of semantic embeddings
        Returns:
            phase: Tensor [B, k, phase_dim] of phase vectors with values in [-1, 1]
        """
        # Apply MLP transformation
        phase_raw = self.mlp(x)
        
        # Apply tanh with diversity scaling
        phase = torch.tanh(phase_raw * self.phase_diversity)
        
        return phase


def contrastive_loss(phase_vectors, semantic_categories, temperature=0.5):
    """
    Calculate contrastive loss to encourage similar phase patterns for similar semantics.
    
    Args:
        phase_vectors: Tensor of shape [B, k, phase_dim]
        semantic_categories: Tensor of shape [B, k] containing category IDs
        temperature: Scaling factor for similarity scores
    
    Returns:
        loss: Contrastive loss value
    """
    # Normalize phase vectors
    phase_norm = F.normalize(phase_vectors, p=2, dim=2)
    
    # Calculate pairwise cosine similarity
    similarity = torch.bmm(phase_norm, phase_norm.transpose(1, 2))
    
    # Create mask of positive pairs (same semantic category)
    pos_mask = (semantic_categories.unsqueeze(2) == semantic_categories.unsqueeze(1))
    
    # Create mask for valid comparisons (exclude self-comparisons)
    diag_mask = ~torch.eye(pos_mask.size(1), dtype=torch.bool, 
                         device=pos_mask.device).unsqueeze(0)
    
    # Apply temperature scaling
    similarity = similarity / temperature
    
    # InfoNCE-style contrastive loss
    exp_sim = torch.exp(similarity)
    
    # For each token, sum similarities with positives (same category)
    pos_sim = torch.sum(exp_sim * pos_mask * diag_mask, dim=2)
    
    # Sum of all similarities (for normalization)
    total_sim = torch.sum(exp_sim * diag_mask, dim=2)
    
    # Calculate loss (-log of normalized positive similarities)
    eps = 1e-8  # For numerical stability
    loss = -torch.log(pos_sim / (total_sim + eps))
    
    # Average over all tokens
    return loss.mean()


class PhaseAnalyzer:
    """
    Tools for analyzing phase vector properties and relationships.
    
    This class provides methods to:
    - Calculate phase similarity metrics
    - Compare intra-category vs. inter-category similarities
    - Generate visualizations of phase relationships
    """
    
    @staticmethod
    def compute_similarity_matrix(phase_vectors):
        """
        Compute cosine similarity matrix between phase vectors.
        
        Args:
            phase_vectors: Tensor of shape [B, k, phase_dim]
            
        Returns:
            similarity: Tensor of shape [B, k, k] with pairwise cosine similarities
        """
        # Normalize phase vectors
        phase_norm = F.normalize(phase_vectors, p=2, dim=2)
        
        # Calculate pairwise cosine similarity
        similarity = torch.bmm(phase_norm, phase_norm.transpose(1, 2))
        
        return similarity
    
    @staticmethod
    def compute_category_similarities(phase_vectors, categories):
        """
        Compute intra-category and inter-category similarities.
        
        Args:
            phase_vectors: Tensor of shape [B, k, phase_dim]
            categories: Tensor of shape [B, k] containing category IDs
            
        Returns:
            dict containing intra_sim, inter_sim, and contrast metrics
        """
        # Compute similarity matrix
        similarity = PhaseAnalyzer.compute_similarity_matrix(phase_vectors)
        
        # Create masks for intra-category and inter-category pairs
        intra_mask = (categories.unsqueeze(2) == categories.unsqueeze(1))
        inter_mask = ~intra_mask
        
        # Create mask to exclude self-comparisons
        diag_mask = ~torch.eye(intra_mask.size(1), dtype=torch.bool, 
                             device=intra_mask.device).unsqueeze(0)
        
        # Apply masks
        intra_mask = intra_mask & diag_mask
        inter_mask = inter_mask & diag_mask
        
        # Calculate mean similarities
        intra_sim = torch.sum(similarity * intra_mask, dim=(1, 2)) / torch.sum(intra_mask, dim=(1, 2))
        inter_sim = torch.sum(similarity * inter_mask, dim=(1, 2)) / torch.sum(inter_mask, dim=(1, 2))
        
        # Calculate contrast (intra - inter)
        contrast = intra_sim - inter_sim
        
        return {
            'intra_similarity': intra_sim,
            'inter_similarity': inter_sim,
            'contrast': contrast
        }
