"""
Feedback loop mechanisms for the Synthetic Thalamus

This module implements various feedback mechanisms to create recurrent connections
between the workspace and the thalamus, inspired by biological thalamo-cortical circuits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Union


class WorkspaceToThalamusFeedback(nn.Module):
    """
    Implements a feedback connection from the workspace to the thalamus.
    
    This allows the thalamus to adjust its salience scoring based on the
    current state of the workspace, creating a dynamic attentional loop.
    """
    
    def __init__(
        self, 
        workspace_dim: int, 
        thalamus_dim: int, 
        feedback_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_gating: bool = True
    ):
        """
        Initialize the feedback module.
        
        Args:
            workspace_dim: Dimension of workspace representations
            thalamus_dim: Dimension of thalamus input features
            feedback_dim: Dimension of the feedback signal
            num_heads: Number of attention heads for feedback
            dropout: Dropout probability
            use_gating: Whether to use a gating mechanism to control feedback strength
        """
        super().__init__()
        
        self.workspace_dim = workspace_dim
        self.thalamus_dim = thalamus_dim
        self.feedback_dim = feedback_dim
        self.use_gating = use_gating
        
        # Project workspace state to feedback signal
        self.workspace_projector = nn.Sequential(
            nn.Linear(workspace_dim, feedback_dim * 2),
            nn.LayerNorm(feedback_dim * 2),
            nn.GELU(),
            nn.Linear(feedback_dim * 2, feedback_dim),
            nn.LayerNorm(feedback_dim)
        )
        
        # Project feedback signal to the same space as thalamus features
        self.feedback_projector = nn.Linear(feedback_dim, thalamus_dim)
        
        # Multi-head attention for applying feedback
        self.attention = nn.MultiheadAttention(
            embed_dim=thalamus_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Optional gating mechanism to control feedback strength
        if use_gating:
            self.gate_network = nn.Sequential(
                nn.Linear(workspace_dim, feedback_dim),
                nn.LayerNorm(feedback_dim),
                nn.GELU(),
                nn.Linear(feedback_dim, 1),
                nn.Sigmoid()
            )
    
    def forward(
        self, 
        workspace_state: torch.Tensor, 
        thalamus_inputs: torch.Tensor,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate feedback signal from workspace state and apply it to thalamus inputs.
        
        Args:
            workspace_state: Tensor of workspace state [B, k, workspace_dim] or [B, workspace_dim]
            thalamus_inputs: Tensor of thalamus inputs [B, N, thalamus_dim]
            return_attention: Whether to return attention weights
            
        Returns:
            modified_inputs: Thalamus inputs modified by feedback [B, N, thalamus_dim]
            attention_weights: Optional attention weights [B, N, N]
        """
        # Handle different input shapes for workspace state
        if workspace_state.dim() == 2:
            # Single pooled representation [B, workspace_dim]
            batch_size = workspace_state.size(0)
            workspace_state = workspace_state.unsqueeze(1)  # [B, 1, workspace_dim]
        else:
            # Multiple token representations [B, k, workspace_dim]
            batch_size = workspace_state.size(0)
            # Pool the workspace state for a compact representation
            workspace_state = workspace_state.mean(dim=1, keepdim=True)  # [B, 1, workspace_dim]
        
        # Generate feedback signal
        feedback = self.workspace_projector(workspace_state)  # [B, 1, feedback_dim]
        
        # Project feedback to thalamus dimension
        feedback = self.feedback_projector(feedback)  # [B, 1, thalamus_dim]
        
        # Calculate gating factor if needed
        if self.use_gating:
            gate = self.gate_network(workspace_state.squeeze(1))  # [B, 1]
            gate = gate.unsqueeze(-1)  # [B, 1, 1]
        else:
            gate = torch.ones(batch_size, 1, 1, device=workspace_state.device)
        
        # Apply attention between feedback and thalamus inputs
        # Using feedback as query and thalamus inputs as keys/values
        attn_output, attn_weights = self.attention(
            query=feedback,
            key=thalamus_inputs,
            value=thalamus_inputs
        )
        
        # Expand the feedback to match thalamus input shape for residual connection
        feedback_expanded = feedback.expand(-1, thalamus_inputs.size(1), -1)
        
        # Apply gated feedback
        modified_inputs = thalamus_inputs + gate * (attn_output - feedback_expanded)
        
        if return_attention:
            return modified_inputs, attn_weights
        else:
            return modified_inputs


class RecurrentThalamusWorkspace(nn.Module):
    """
    Implements a recurrent thalamus-workspace circuit with feedback loops.
    
    This module manages the iterative processing between thalamus and workspace,
    allowing for multiple steps of information refinement before producing the final output.
    """
    
    def __init__(
        self,
        thalamus: nn.Module,
        workspace: nn.Module,
        feedback: WorkspaceToThalamusFeedback,
        max_iterations: int = 3,
        adaptive_iterations: bool = False,
        halt_threshold: float = 0.05
    ):
        """
        Initialize the recurrent thalamus-workspace circuit.
        
        Args:
            thalamus: SyntheticThalamus module
            workspace: EnhancedWorkspace module
            feedback: WorkspaceToThalamusFeedback module
            max_iterations: Maximum number of recurrent iterations
            adaptive_iterations: Whether to use adaptive iteration termination
            halt_threshold: Threshold for halting iterations (if adaptive)
        """
        super().__init__()
        
        self.thalamus = thalamus
        self.workspace = workspace
        self.feedback = feedback
        
        self.max_iterations = max_iterations
        self.adaptive_iterations = adaptive_iterations
        self.halt_threshold = halt_threshold
        
        # For adaptive iterations, we need a way to determine when to stop
        if adaptive_iterations:
            self.halt_network = nn.Sequential(
                nn.Linear(workspace.d_model, workspace.d_model // 2),
                nn.LayerNorm(workspace.d_model // 2),
                nn.GELU(),
                nn.Linear(workspace.d_model // 2, 1),
                nn.Sigmoid()
            )
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        task_ids: Optional[torch.Tensor] = None,
        return_intermediates: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Process inputs through the recurrent thalamus-workspace circuit.
        
        Args:
            inputs: Tensor of shape [B, N, D] with encoded feature tokens
            task_ids: Optional tensor of shape [B] with task identifiers
            return_intermediates: Whether to return intermediate states
            
        Returns:
            outputs: Final workspace outputs
            pooled: Pooled representation
            intermediates: Optional dictionary of intermediate states
        """
        batch_size = inputs.size(0)
        
        # Initialize workspace state
        workspace_state = None
        
        # Keep track of intermediate states if needed
        intermediates = {
            'thalamus_outputs': [],
            'workspace_outputs': [],
            'feedback_gates': [],
            'halt_scores': []
        }
        
        # Recurrent processing loop
        for i in range(self.max_iterations):
            # Apply feedback if we have workspace state
            if workspace_state is not None:
                inputs_modified = self.feedback(workspace_state, inputs)
            else:
                inputs_modified = inputs
            
            # Process through thalamus
            thalamus_output = self.thalamus(inputs_modified, task_ids)
            
            # Process through workspace
            workspace_output, pooled = self.workspace(thalamus_output)
            
            # Store intermediate states if requested
            if return_intermediates:
                intermediates['thalamus_outputs'].append(thalamus_output.detach())
                intermediates['workspace_outputs'].append(workspace_output.detach())
                
                # Store feedback gate if available
                if hasattr(self.feedback, 'gate_network') and i > 0:
                    gate = self.feedback.gate_network(workspace_state)
                    intermediates['feedback_gates'].append(gate.detach())
            
            # Update workspace state
            workspace_state = pooled
            
            # Check for early stopping if using adaptive iterations
            if self.adaptive_iterations and i > 0:
                halt_score = self.halt_network(pooled).mean()
                
                if return_intermediates:
                    intermediates['halt_scores'].append(halt_score.detach())
                
                if halt_score > self.halt_threshold:
                    break
        
        if return_intermediates:
            return workspace_output, pooled, intermediates
        else:
            return workspace_output, pooled


class CrossModalFeedback(nn.Module):
    """
    Implements cross-modal feedback between different thalamus modules.
    
    This allows for information sharing between different modalities (e.g., text and vision),
    creating a more integrated multi-modal processing system.
    """
    
    def __init__(
        self,
        modal_dims: Dict[str, int],
        feedback_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize the cross-modal feedback module.
        
        Args:
            modal_dims: Dictionary mapping modality names to feature dimensions
            feedback_dim: Dimension of the feedback signal
            num_heads: Number of attention heads for feedback
            dropout: Dropout probability
        """
        super().__init__()
        
        self.modal_dims = modal_dims
        self.feedback_dim = feedback_dim
        
        # Create projectors for each modality
        self.encoders = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, feedback_dim),
                nn.LayerNorm(feedback_dim),
                nn.GELU()
            ) for name, dim in modal_dims.items()
        })
        
        # Create projectors back to each modality
        self.decoders = nn.ModuleDict({
            name: nn.Linear(feedback_dim, dim)
            for name, dim in modal_dims.items()
        })
        
        # Cross-modal attention
        self.cross_attention = nn.ModuleDict({
            f"{src}_to_{tgt}": nn.MultiheadAttention(
                embed_dim=feedback_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            ) for src in modal_dims for tgt in modal_dims if src != tgt
        })
        
        # Gating networks to control cross-modal influence
        self.gates = nn.ModuleDict({
            f"{src}_to_{tgt}": nn.Sequential(
                nn.Linear(feedback_dim * 2, feedback_dim),
                nn.LayerNorm(feedback_dim),
                nn.GELU(),
                nn.Linear(feedback_dim, 1),
                nn.Sigmoid()
            ) for src in modal_dims for tgt in modal_dims if src != tgt
        })
    
    def forward(self, modal_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply cross-modal feedback between different modalities.
        
        Args:
            modal_inputs: Dictionary mapping modality names to input tensors
            
        Returns:
            modified_inputs: Dictionary of modified inputs with cross-modal feedback
        """
        # Encode all modalities to a common space
        encoded = {
            name: self.encoders[name](tensor)
            for name, tensor in modal_inputs.items()
        }
        
        # Process cross-modal attention
        cross_modal_features = {}
        for tgt_name in self.modal_dims:
            # Skip if this modality isn't in the inputs
            if tgt_name not in modal_inputs:
                continue
                
            aggregated = []
            for src_name in self.modal_dims:
                # Skip self-attention and missing modalities
                if src_name == tgt_name or src_name not in modal_inputs:
                    continue
                
                # Cross-modal attention from source to target
                attn_key = f"{src_name}_to_{tgt_name}"
                attn_output, _ = self.cross_attention[attn_key](
                    query=encoded[tgt_name],
                    key=encoded[src_name],
                    value=encoded[src_name]
                )
                
                # Calculate gate based on both source and target features
                # Concatenate target and attention output for gating
                concat_features = torch.cat([encoded[tgt_name], attn_output], dim=-1)
                gate = self.gates[attn_key](concat_features)
                
                # Apply gated cross-modal feature
                aggregated.append(gate * attn_output)
            
            # Combine all cross-modal features if we have any
            if aggregated:
                cross_modal_sum = sum(aggregated) / len(aggregated)
                cross_modal_features[tgt_name] = cross_modal_sum
            else:
                cross_modal_features[tgt_name] = encoded[tgt_name]
        
        # Decode back to original modality spaces
        modified_inputs = {
            name: modal_inputs[name] + self.decoders[name](cross_modal_features[name])
            for name in cross_modal_features
        }
        
        return modified_inputs


class HierarchicalThalamus(nn.Module):
    """
    Implements a hierarchical version of the synthetic thalamus with multiple layers.
    
    This creates a stacked architecture where each layer processes progressively
    more abstract features, similar to the hierarchical organization of the biological thalamus.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: List[int],
        k_values: List[int],
        phase_dim: int,
        task_dim: int,
        num_tasks: int,
        num_layers: int = 2,
        feedback_connections: bool = True,
        phase_configs: Optional[List[Dict]] = None
    ):
        """
        Initialize the hierarchical thalamus.
        
        Args:
            d_model: Dimension of input feature vectors
            n_heads: List of attention heads for each layer
            k_values: List of top-k values for each layer
            phase_dim: Dimensionality of the phase tag
            task_dim: Dimensionality of task conditioning
            num_tasks: Number of distinct tasks
            num_layers: Number of hierarchical layers
            feedback_connections: Whether to include feedback between layers
            phase_configs: Optional list of configs for phase generators
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.feedback_connections = feedback_connections
        
        # Validate inputs
        assert len(n_heads) == num_layers, "Must provide n_heads for each layer"
        assert len(k_values) == num_layers, "Must provide k_values for each layer"
        
        # Set default phase configurations if not provided
        if phase_configs is None:
            phase_configs = [
                {
                    'hidden_dims': [128, 64],
                    'activation': 'gelu',
                    'phase_diversity': 2.0,
                    'use_layer_norm': True
                } for _ in range(num_layers)
            ]
        
        # Create thalamus layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            from core.thalamus import SyntheticThalamus
            
            layer = SyntheticThalamus(
                d_model=d_model,
                n_heads=n_heads[i],
                k=k_values[i],
                phase_dim=phase_dim,
                task_dim=task_dim,
                num_tasks=num_tasks,
                **phase_configs[i]
            )
            self.layers.append(layer)
        
        # Create feedback connections between layers if enabled
        if feedback_connections and num_layers > 1:
            self.feedbacks = nn.ModuleList()
            for i in range(num_layers - 1):
                # Feedback from layer i+1 to layer i
                feedback = nn.Sequential(
                    nn.Linear(d_model + phase_dim, d_model),
                    nn.LayerNorm(d_model),
                    nn.GELU(),
                    nn.Linear(d_model, d_model)
                )
                self.feedbacks.append(feedback)
    
    def forward(
        self, 
        x: torch.Tensor, 
        task_id: torch.Tensor, 
        context: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Process inputs through the hierarchical thalamus.
        
        Args:
            x: Tensor of shape [B, N, D] with encoded feature tokens
            task_id: Long tensor of shape [B] containing task identifiers
            context: Optional tensor [B, M, D] for additional workspace feedback
            
        Returns:
            outputs: List of outputs from each layer [B, k_i, D + phase_dim]
        """
        layer_inputs = x
        layer_outputs = []
        
        # Forward pass through each layer
        for i, layer in enumerate(self.layers):
            # Process through the current layer
            layer_output = layer(layer_inputs, task_id, context)
            layer_outputs.append(layer_output)
            
            # Prepare inputs for the next layer
            if i < self.num_layers - 1:
                if self.feedback_connections and i > 0:
                    # Apply feedback from previous layer's output
                    feedback = self.feedbacks[i-1](layer_outputs[i-1])
                    layer_inputs = x + feedback
                else:
                    # No feedback, just use original inputs
                    layer_inputs = x
        
        return layer_outputs
