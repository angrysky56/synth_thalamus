# core/thalamus.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.phase_generator import EnhancedSemanticPhaseGenerator

# For backward compatibility
class SemanticPhaseGenerator(nn.Module):
    def __init__(self, d_model, phase_dim, hidden_dim=64, phase_diversity=1.0):
        """
        Maps semantic embeddings to phase vectors.
        
        Args:
            d_model: Dimensionality of the semantic embedding.
            phase_dim: Desired phase dimensionality.
            hidden_dim: Number of hidden units in the MLP.
            phase_diversity: Scaling factor to encourage phase diversity.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, phase_dim)
        )
        # Additional scaling factor for phase diversity control
        self.phase_diversity = phase_diversity

    def forward(self, x):
        """
        Args:
            x: Tensor of semantic embeddings with shape [B, k, d_model].
        Returns:
            phase: Tensor of shape [B, k, phase_dim] with values in [-1, 1],
                   modulated by the learned non-linearity.
        """
        phase_raw = self.mlp(x)  # [B, k, phase_dim]
        # Apply a non-linear activation (e.g., scaled tanh) to restrict range and encourage diversity.
        phase = torch.tanh(phase_raw * self.phase_diversity)
        return phase

class SyntheticThalamus(nn.Module):
    def __init__(self, d_model, n_heads=4, k=32, phase_dim=16, task_dim=64, num_tasks=10, 
                 phase_diversity=1.0, hidden_dims=[128, 64], activation='gelu', use_layer_norm=True):
        """
        Args:
            d_model: Dimension of input feature vectors.
            n_heads: Number of attention heads.
            k: Top-K tokens to gate.
            phase_dim: Dimensionality of the phase tag.
            task_dim: Dimensionality of task conditioning.
            num_tasks: Number of distinct tasks.
            phase_diversity: Controls the amount of variation between token phases (0-1).
            hidden_dims: List of hidden layer dimensions for phase generator.
            activation: Activation function type ('gelu', 'leaky_relu', 'silu', 'prelu', 'relu').
            use_layer_norm: Whether to use layer normalization in phase generator.
        """
        super().__init__()
        self.k = k
        self.d_model = d_model
        self.phase_dim = phase_dim
        self.phase_diversity = phase_diversity

        # Task embedding to condition salience scoring.
        self.task_embed = nn.Embedding(num_tasks, task_dim)
        self.task_proj = nn.Linear(task_dim, d_model)

        # Multihead Attention for salience scoring.
        self.scorer = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        # Use the enhanced phase generator
        self.phase_generator = EnhancedSemanticPhaseGenerator(
            d_model=d_model, 
            phase_dim=phase_dim, 
            hidden_dims=hidden_dims, 
            activation=activation,
            phase_diversity=phase_diversity,
            use_layer_norm=use_layer_norm
        )
        
        # For backward compatibility, keep the phase_freqs and phase_proj
        # These won't be used with the new phase generator but keep them
        # to avoid breaking existing code that might reference them
        self.phase_freqs = nn.Parameter(torch.randn(phase_dim) * 0.1)
        self.phase_proj = nn.Linear(d_model, phase_dim)

    def forward(self, x, task_id, context=None):
        """
        Args:
            x: Tensor of shape [B, N, D] with encoded feature tokens.
            task_id: Long tensor of shape [B] containing task identifiers.
            context: Optional tensor [B, M, D] for additional workspace feedback.
        Returns:
            gated: Tensor of shape [B, k, D + phase_dim] with gated tokens and phase tags.
            indices: Tensor of shape [B, k] containing original indices of selected tokens.
        """
        B, N, D = x.size()

        # Embed task ID and project into d_model space.
        task_embedding = self.task_embed(task_id)          # [B, task_dim]
        task_embedding = self.task_proj(task_embedding)      # [B, d_model]
        # Expand task embedding across tokens and add.
        task_embedding = task_embedding.unsqueeze(1).expand(-1, N, -1)  # [B, N, d_model]
        x_combined = x + task_embedding

        # Use context if provided; otherwise, self-attention.
        if context is not None:
            x_attn, _ = self.scorer(x_combined, context, context)
        else:
            x_attn, _ = self.scorer(x_combined, x_combined, x_combined)

        # Salience scoring: here we take the norm of each token.
        scores = x_attn.norm(dim=-1)  # [B, N]
        # Select the top-k tokens per sample.
        _, topk_indices = scores.topk(self.k, dim=1)  # [B, k]
        # Gather the corresponding tokens.
        gated = torch.gather(x_attn, 1, topk_indices.unsqueeze(-1).expand(-1, -1, D))

        # Generate phase tags directly from the gated semantic content
        phase = self.phase_generator(gated)  # [B, k, phase_dim]
        
        # Concatenate the original gated token with its phase tag.
        gated = torch.cat([gated, phase], dim=-1)  # [B, k, D + phase_dim]
        return gated, topk_indices