# core/adapters.py
import torch
import torch.nn as nn
import torchvision.models as models

class ImageAdapter(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        # Use a pretrained ResNet18 as a simple image encoder.
        self.cnn = models.resnet18(pretrained=True)
        # Replace final layer with a projection to d_model.
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, d_model)

    def forward(self, x):
        # x: [B, 3, H, W]
        return self.cnn(x)

class TextAdapter(nn.Module):
    def __init__(self, vocab_size, d_model=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # x: [B, T] token IDs; return average embedding.
        embeddings = self.embedding(x)  # [B, T, d_model]
        return embeddings.mean(dim=1)