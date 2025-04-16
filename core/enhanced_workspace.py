import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PhaseSimilarityTransformerLayer(nn.Module):
    """
    Transformer layer with phase similarity attention bias.
    
    This layer extends the standard transformer attention mechanism by incorporating
    phase similarity information to bias the attention weights. Tokens with similar
    phase vectors will attend more strongly to each other.
    """
    def __init__(self, d_model, nhead, phase_dim, dropout=0.1, initial_phase_scale=1.0):
        """
        Initialize the phase similarity transformer layer.
        
        Args:
            d_model: Dimension of the model (input and output)
            nhead: Number of attention heads
            phase_dim: Dimensionality of the phase vectors
            dropout: Dropout probability for attention weights
            initial_phase_scale: Initial value for the phase scale parameter
        """
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = nn.Dropout(dropout)
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"
        
        # Learned linear projections for q, k, and v.
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Scale factor for phase bias: a learnable parameter.
        self.phase_scale = nn.Parameter(torch.tensor(initial_phase_scale))
        self.phase_dim = phase_dim

    def forward(self, content, phase):
        """
        Forward pass through the transformer layer.
        
        Args:
            content: Tensor [B, k, d_model] (token representations)
            phase: Tensor [B, k, phase_dim] (phase tags for each token)
        
        Returns:
            Tuple of:
            - transformed_content: Tensor [B, k, d_model] representing the transformed content
            - attn_weights: Tensor [B, nhead, k, k] containing attention weights for visualization
        """
        B, k, d_model = content.shape

        # Project q, k, v from the content.
        q = self.q_proj(content)  # [B, k, d_model]
        k_ = self.k_proj(content) # [B, k, d_model]
        v = self.v_proj(content)  # [B, k, d_model]

        # Reshape for multihead attention: [B, k, nhead, head_dim] then transpose to [B, nhead, k, head_dim]
        def shape(x):
            return x.view(B, k, self.nhead, self.head_dim).transpose(1, 2)
        q = shape(q)
        k_ = shape(k_)
        v = shape(v)

        # Compute raw attention logits.
        q = q / math.sqrt(self.head_dim)
        attn_logits = torch.matmul(q, k_.transpose(-2, -1))  # [B, nhead, k, k]

        # Compute phase similarity bias.
        # Normalize phase vectors along the phase_dim.
        phase_norm = F.normalize(phase, p=2, dim=-1)  # [B, k, phase_dim]
        # Compute cosine similarity for each pair of tokens.
        phase_bias = torch.bmm(phase_norm, phase_norm.transpose(1, 2))  # [B, k, k]
        # Expand phase_bias to match attention heads.
        phase_bias = phase_bias.unsqueeze(1)  # [B, 1, k, k]
        # Scale the bias and add to the attention logits.
        attn_logits = attn_logits + self.phase_scale * phase_bias

        # Apply softmax to get attention weights.
        attn_weights = F.softmax(attn_logits, dim=-1)  # [B, nhead, k, k]
        attn_weights = self.dropout(attn_weights)

        # Compute attention output.
        attn_output = torch.matmul(attn_weights, v)  # [B, nhead, k, head_dim]
        # Combine heads.
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, k, d_model)
        out = self.out_proj(attn_output)
        return out, attn_weights  # Return attention weights for visualization

class EnhancedWorkspace(nn.Module):
    """
    Enhanced workspace with phase similarity attention.
    
    This module processes tokens with their associated phase tags using a stack of
    phase-aware transformer layers. It leverages phase similarity to bias attention
    weights, helping the model identify which tokens should be processed together.
    """
    def __init__(
        self, 
        input_dim, 
        hidden_dim, 
        output_dim, 
        nhead=4, 
        phase_dim=16, 
        num_layers=2, 
        dropout=0.1,
        activation='relu',
        initial_phase_scale=1.0
    ):
        """
        Initialize the enhanced workspace.
        
        Args:
            input_dim: Dimension of the content part (without phase).
            hidden_dim: Internal dimension for feed-forward layers.
            output_dim: Output dimension (e.g., classification logits).
            nhead: Number of attention heads.
            phase_dim: Dimensionality of phase tags.
            num_layers: Number of transformer layers to stack.
            dropout: Dropout probability.
            activation: Activation function for feed-forward network ('relu', 'gelu').
            initial_phase_scale: Initial value for phase scale parameter.
        """
        super().__init__()
        # The total dimension is input_dim (content) + phase_dim.
        self.d_model = input_dim  # We assume the content projection maintains this dimension.
        self.phase_dim = phase_dim
        
        # Stack multiple transformer layers
        self.phase_layers = nn.ModuleList([
            PhaseSimilarityTransformerLayer(
                self.d_model, 
                nhead, 
                phase_dim, 
                dropout=dropout,
                initial_phase_scale=initial_phase_scale
            )
            for _ in range(num_layers)
        ])
        
        # Feed-forward network after transformer layers
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, hidden_dim),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        
        # Simple classification head
        self.fc = nn.Linear(self.d_model, output_dim)
        
        # Store attention weights for visualization
        self.attention_weights = None

    def forward(self, x_with_phase):
        """
        Process input tokens with their phase tags.
        
        Args:
            x_with_phase: Tensor [B, k, D + phase_dim]
                where the first D components are content and the remaining are phase.
        
        Returns:
            Tuple of:
            - output: Tensor [B, output_dim] containing output logits
            - pooled: Tensor [B, d_model] containing pooled representations
        """
        # Split content and phase
        content_dim = x_with_phase.size(-1) - self.phase_dim
        content = x_with_phase[..., :content_dim]  # [B, k, content_dim]
        phase = x_with_phase[..., content_dim:]    # [B, k, phase_dim]
        
        # Process with phase-aware transformer layers
        self.attention_weights = []
        for layer in self.phase_layers:
            # Apply transformer layer with residual connection and normalization
            layer_out, attn_weights = layer(content, phase)
            content = self.norm1(content + layer_out)
            
            # Apply feed-forward network with residual connection and normalization
            ffn_out = self.ffn(content)
            content = self.norm2(content + ffn_out)
            
            # Store attention weights for visualization
            self.attention_weights.append(attn_weights)
        
        # For simplicity, aggregate using mean pooling
        pooled = content.mean(dim=1)  # [B, D]
        output = self.fc(pooled)
        
        return output, pooled
