# train.py
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from core.thalamus import SyntheticThalamus

# Import the enhanced workspace
from core.enhanced_workspace import EnhancedWorkspace

# Original Workspace kept for backwards compatibility
class Workspace(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x: [B, k, input_dim]
        x_trans = self.transformer(x)  # [B, k, input_dim]
        x_mean = x_trans.mean(dim=1)   # Aggregate over tokens
        output = self.fc(x_mean)
        return output, x_mean

class SyntheticThalamusModel(pl.LightningModule):
    def __init__(self, d_model=128, output_dim=10, num_tasks=10, vocab_size=10000, use_enhanced_workspace=True):
        super().__init__()
        # Initialize a basic encoder (here, a linear layer for demonstration).
        self.encoder = nn.Linear(d_model, d_model)
        
        # Initialize the synthetic thalamus.
        self.phase_dim = 16
        self.thalamus = SyntheticThalamus(
            d_model=d_model, n_heads=4, k=32, phase_dim=self.phase_dim, task_dim=64, num_tasks=num_tasks
        )
        
        # Workspace to process gated tokens - choose between simple and enhanced
        if use_enhanced_workspace:
            self.workspace = EnhancedWorkspace(
                input_dim=d_model, 
                hidden_dim=256, 
                output_dim=output_dim,
                nhead=4,
                phase_dim=self.phase_dim,
                num_layers=2
            )
        else:
            self.workspace = Workspace(
                input_dim=d_model + self.phase_dim, 
                hidden_dim=256, 
                output_dim=output_dim
            )
            
        self.loss_fn = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, x, task_id, context=None):
        # x: [B, N, d_model]
        x_enc = self.encoder(x)
        gated, _ = self.thalamus(x_enc, task_id, context)
        output, _ = self.workspace(gated)
        return output

    def training_step(self, batch, batch_idx):
        # Expect batch: (x, task_id, target)
        x, task_id, target = batch
        logits = self(x, task_id)
        loss = self.loss_fn(logits, target)
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, task_id, target = batch
        logits = self(x, task_id)
        loss = self.loss_fn(logits, target)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def main():
    parser = argparse.ArgumentParser(description='Train Synthetic Thalamus Model')
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--enhanced_workspace', action='store_true', 
                        help='Use the enhanced workspace with phase similarity attention')
    
    args = parser.parse_args()
    
    # Example dummy data setup
    class DummyDataModule(pl.LightningDataModule):
        def __init__(self, batch_size=32):
            super().__init__()
            self.batch_size = batch_size
            
        def setup(self, stage=None):
            # Create dummy data
            self.train_x = torch.randn(1000, 100, 128)  # 1000 samples, 100 tokens, 128 dim
            self.train_task_ids = torch.randint(0, 10, (1000,))
            self.train_targets = torch.randint(0, 10, (1000,))
            
            self.val_x = torch.randn(200, 100, 128)
            self.val_task_ids = torch.randint(0, 10, (200,))
            self.val_targets = torch.randint(0, 10, (200,))
            
        def train_dataloader(self):
            train_data = list(zip(self.train_x, self.train_task_ids, self.train_targets))
            return torch.utils.data.DataLoader(train_data, batch_size=self.batch_size)
            
        def val_dataloader(self):
            val_data = list(zip(self.val_x, self.val_task_ids, self.val_targets))
            return torch.utils.data.DataLoader(val_data, batch_size=self.batch_size)
    
    # Initialize model with enhanced workspace if requested
    model = SyntheticThalamusModel(
        d_model=args.d_model,
        use_enhanced_workspace=args.enhanced_workspace
    )
    data_module = DummyDataModule(batch_size=args.batch_size)
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else None,
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    print("Training completed successfully!")

if __name__ == '__main__':
    # Example: initialize and run a dummy training step.
    model = SyntheticThalamusModel()
    x_dummy = torch.randn(4, 100, 128)  # Batch of 4, 100 tokens each
    task_id_dummy = torch.randint(0, 10, (4,))
    target_dummy = torch.randint(0, 10, (4,))
    # Forward pass
    logits = model(x_dummy, task_id_dummy)
    print("Logits shape:", logits.shape)
    
    # Uncomment to run the main training loop
    # main()