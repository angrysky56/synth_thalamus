"""
Ollama-Enhanced Phase Similarity Demo

This script demonstrates the enhanced workspace with phase similarity attention bias
using real embeddings from Ollama models instead of random tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import ollama

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from core.thalamus import SyntheticThalamus
from core.enhanced_workspace import EnhancedWorkspace

# Define a simple workspace class (instead of importing from train.py)
class SimpleWorkspace(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [B, k, input_dim]
        x_mean = x.mean(dim=1)   # Aggregate over tokens
        hidden = F.relu(self.fc1(x_mean))
        output = self.fc2(hidden)
        return output, x_mean

def get_ollama_embeddings(texts, model_name="mxbai-embed-large:latest"):
    """
    Get embeddings for a list of texts using Ollama.
    """
    embeddings = []
    
    print(f"Using Ollama model: {model_name}")
    for i, text in enumerate(texts):
        try:
            response = ollama.embeddings(model=model_name, prompt=text)
            embedding = response['embedding']
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error with text '{text}': {e}")
            # Try with fallback model if primary fails
            if model_name != "phi4-mini:latest":
                print("Attempting with fallback model phi4-mini...")
                try:
                    response = ollama.embeddings(model="phi4-mini:latest", prompt=text)
                    embedding = response['embedding']
                    embeddings.append(embedding)
                except Exception as e2:
                    print(f"Fallback also failed: {e2}")
                    # Use a random embedding as last resort
                    embeddings.append(np.random.randn(4096).tolist())
            else:
                # Use a random embedding as last resort
                embeddings.append(np.random.randn(4096).tolist())
        
        # Progress indicator
        if (i+1) % 5 == 0:
            print(f"Processed {i+1}/{len(texts)} texts")
            
    return torch.tensor(embeddings)

def create_semantic_data(d_model=128):
    """Create data with semantic meaning using Ollama embeddings."""
    # Define groups of semantically similar texts
    text_groups = [
        # Technology group
        ["computer", "laptop", "smartphone", "internet", "software", "hardware", "algorithm", "code"],
        # Nature group
        ["forest", "mountain", "river", "ocean", "tree", "flower", "animal", "ecosystem"],
        # Food group
        ["pizza", "hamburger", "pasta", "salad", "dessert", "breakfast", "dinner", "chef"],
        # Transport group
        ["car", "plane", "train", "bicycle", "subway", "ship", "rocket", "helicopter"]
    ]
    
    # Flatten the groups
    all_texts = [item for group in text_groups for item in group]
    
    print(f"Getting embeddings for {len(all_texts)} texts using Ollama...")
    # Get embeddings from Ollama
    try:
        embeddings = get_ollama_embeddings(all_texts)
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Falling back to random embeddings")
        # Fallback to random embeddings if Ollama is not available
        embeddings = torch.randn(len(all_texts), 4096)
    
    # Print the original embedding dimension
    print(f"Original embedding dimension: {embeddings.shape[1]}")
    
    # Resize embeddings to the desired dimension
    if embeddings.shape[1] != d_model:
        projection = nn.Linear(embeddings.shape[1], d_model)
        embeddings = projection(embeddings)
    
    # Add batch dimension
    embeddings = embeddings.unsqueeze(0)  # [1, num_tokens, d_model]
    
    # Create labels corresponding to the group indices
    group_sizes = [len(group) for group in text_groups]
    labels = torch.tensor([[i for i, size in enumerate(group_sizes) for _ in range(size)]])
    
    # Create task IDs (just 0 for simplicity)
    task_ids = torch.zeros(1, dtype=torch.long)
    
    return embeddings, task_ids, labels, all_texts

def visualize_attention_patterns(model, tokens, task_ids, token_labels, enhanced=True):
    """Visualize attention patterns in the workspace."""
    with torch.no_grad():
        # Encode tokens
        encoded = model.encoder(tokens)
        
        # Apply thalamus
        gated = model.thalamus(encoded, task_ids)
        
        # Calculate D (content dimension)
        D = tokens.size(-1)
        
        # Forward through workspace
        if enhanced:
            # Enhanced workspace stores attention weights
            logits, _ = model.workspace(gated)
            attention_weights = model.workspace.attention_weights[0]  # Get first layer's weights
            
            # Average across heads and batch
            attn_avg = attention_weights.mean(dim=1).mean(dim=0).cpu().numpy()
            
            # Extract phase information for visualization
            phase = gated[..., D:].cpu().numpy()[0]  # Take first batch item
        else:
            # Standard workspace doesn't have attention weights
            logits, _ = model.workspace(gated)
            attn_avg = None
            phase = gated[..., D:].cpu().numpy()[0]
        
        # Print phase stats for debugging
        print(f"Phase shape: {phase.shape}")
        print(f"Phase min: {phase.min():.4f}, max: {phase.max():.4f}, mean: {phase.mean():.4f}")
        
        # Calculate phase similarity matrix to verify diversity
        phase_tensor = torch.tensor(phase)
        phase_norm = F.normalize(phase_tensor, p=2, dim=1)
        similarity = torch.mm(phase_norm, phase_norm.t()).cpu().numpy()
        # Diagonal should be 1.0 (self-similarity), off-diagonal should be less than 1.0
        np.fill_diagonal(similarity, 0)  # Zero out diagonal for better visualization
        print(f"Phase similarity (higher=more similar): min={similarity.min():.4f}, max={similarity.max():.4f}, mean={similarity.mean():.4f}")
    
    # Create visualization
    plt.figure(figsize=(16, 12))
    
    # 2x2 subplot layout
    if attn_avg is not None:
        # Plot attention heatmap
        plt.subplot(2, 2, 1)
        im = plt.imshow(attn_avg, cmap='viridis')
        plt.colorbar(im, label='Attention Weight')
        plt.title('Attention Pattern (Enhanced Workspace)')
        
        # Add text labels to the axes
        plt.xticks(range(len(token_labels)), token_labels, rotation=45, ha="right")
        plt.yticks(range(len(token_labels)), token_labels)
        plt.xlabel('Target Token')
        plt.ylabel('Source Token')
    
        # Plot phase similarity matrix
        plt.subplot(2, 2, 2)
        im = plt.imshow(similarity, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(im, label='Cosine Similarity')
        plt.title('Phase Similarity Between Tokens')
        
        # Add text labels to the axes
        plt.xticks(range(len(token_labels)), token_labels, rotation=45, ha="right")
        plt.yticks(range(len(token_labels)), token_labels)
        plt.xlabel('Token')
        plt.ylabel('Token')
    
    # Plot each phase vector as a line
    plt.subplot(2, 2, 3)
    for i in range(min(phase.shape[0], 8)):  # Plot up to 8 tokens
        plt.plot(range(phase.shape[1]), phase[i], label=f'{token_labels[i]}', marker='o')
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel('Phase Dimension')
    plt.ylabel('Phase Value')
    plt.title('Phase Vectors')
    plt.legend()
    
    # Plot phase vectors heatmap
    plt.subplot(2, 2, 4)
    im = plt.imshow(phase, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im, label='Phase Value')
    plt.title('Phase Vectors Heatmap')
    plt.yticks(range(len(token_labels)), token_labels)
    plt.xlabel('Phase Dimension')
    plt.ylabel('Token Index')
    
    plt.tight_layout()
    plt.savefig(f"attention_ollama_{'enhanced' if enhanced else 'standard'}.png", dpi=300)
    plt.close()
    
    print(f"Visualization saved to 'attention_ollama_{'enhanced' if enhanced else 'standard'}.png'")

def main():
    # Parameters
    d_model = 128
    output_dim = 10
    
    # Create semantic data using Ollama
    tokens, task_ids, _, token_texts = create_semantic_data(d_model=d_model)
    
    print("\n=== Demo with semantic tokens from Ollama ===")
    # Initialize thalamus
    thalamus_params = {
        'd_model': d_model,
        'n_heads': 4,
        'k': 8,  # Select top 8 tokens
        'phase_dim': 16,
        'task_dim': 64,
        'num_tasks': 10,
        'phase_diversity': 0.8  # High diversity
    }
    
    # Create models with enhanced workspace
    enhanced_model = torch.nn.Module()
    enhanced_model.encoder = nn.Identity()  # No need to encode, tokens are already embeddings
    enhanced_model.thalamus = SyntheticThalamus(**thalamus_params)
    enhanced_model.workspace = EnhancedWorkspace(
        input_dim=d_model, hidden_dim=256, output_dim=output_dim,
        nhead=4, phase_dim=16, num_layers=2
    )
    
    # Set model to evaluation mode
    enhanced_model.eval()
    
    # Make the phases more visible (increase magnitude of learned frequencies)
    with torch.no_grad():
        # Initialize phase_freqs with larger values for better visualization
        enhanced_model.thalamus.phase_freqs.data = torch.linspace(0.5, 2.0, 16)
    
    # Select top 8 tokens for visualization (the number that will be gated through the thalamus)
    token_labels = token_texts[:8]
    
    print("\nVisualizing enhanced workspace with semantic tokens...")
    visualize_attention_patterns(enhanced_model, tokens, task_ids, token_labels, enhanced=True)
    
    print("\nDemo complete!")

if __name__ == "__main__":
    main()
