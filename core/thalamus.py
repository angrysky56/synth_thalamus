# core/thalamus.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SyntheticThalamus(nn.Module):
    def __init__(self, d_model, n_heads=4, k=32, phase_dim=16, task_dim=64, num_tasks=10, phase_diversity=0.5):
        """
        Args:
            d_model: Dimension of input feature vectors.
            n_heads: Number of attention heads.
            k: Top-K tokens to gate.
            phase_dim: Dimensionality of the phase tag.
            task_dim: Dimensionality of task conditioning.
            num_tasks: Number of distinct tasks.
            phase_diversity: Controls the amount of variation between token phases (0-1).
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

        # Phase generator: projects tokens into a phase space.
        self.phase_proj = nn.Linear(d_model, phase_dim)
        # Learnable frequency scaling for phase modulation.
        self.phase_freqs = nn.Parameter(torch.randn(phase_dim) * 0.1)
        
        # Position-dependent phase offset - adds variation between tokens
        self.position_proj = nn.Linear(d_model, phase_dim)

    def forward(self, x, task_id, context=None):
        """
        Args:
            x: Tensor of shape [B, N, D] with encoded feature tokens.
            task_id: Long tensor of shape [B] containing task identifiers.
            context: Optional tensor [B, M, D] for additional workspace feedback.
        Returns:
            Tensor of shape [B, k, D + phase_dim] with gated tokens and phase tags.
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
        _, topk_indices = scores.topk(self.k, dim=1)
        # Gather the corresponding tokens.
        gated = torch.gather(x_attn, 1, topk_indices.unsqueeze(-1).expand(-1, -1, D))

        # Generate phase tags with increased diversity
        # Base phase projection
        phase_base = self.phase_proj(gated)
        
        # Create position-dependent offsets to increase token variation
        # This ensures different tokens get different phase patterns
        position_offsets = self.position_proj(gated) * self.phase_diversity
        
        # Apply different starting points for different tokens
        # Higher phase_diversity = more variation between tokens
        phase = torch.sin(phase_base * self.phase_freqs + position_offsets)
        
        # Concatenate the original gated token with its phase tag.
        gated = torch.cat([gated, phase], dim=-1)  # [B, k, D + phase_dim]
        return gated