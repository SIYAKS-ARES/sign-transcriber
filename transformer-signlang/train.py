#!/usr/bin/env python3
"""
Transformer Sign Language Classifier - Training Script
------------------------------------------------------
Eğitim scripti: Model eğitimi, validation ve checkpoint kaydetme

Kullanım:
    python train.py

Gereksinimler:
    - data/processed/ altında hazırlanmış veriler
    - config.py'da tanımlı hiperparametreler
    
Çıktılar:
    - checkpoints/best_model.pth
    - checkpoints/last_model.pth
    - logs/training.log
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import json
from datetime import datetime
from pathlib import Path

# Proje root'unu path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import TransformerConfig
from models.transformer_model import TransformerSignLanguageClassifier, print_model_summary


# ==================== DATASET ====================

class SignLanguageDataset(Dataset):
    """
    PyTorch Dataset for Sign Language Keypoints
    
    Args:
        X: Feature array (N, seq_len, 258) - numpy or torch tensor
        y: Label array (N,) - numpy or torch tensor
    """
    
    def __init__(self, X, y):
        if isinstance(X, np.ndarray):
            self.X = torch.FloatTensor(X)
        else:
            self.X = X.float()
        
        if isinstance(y, np.ndarray):
            self.y = torch.LongTensor(y)
        else:
            self.y = y.long()
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==================== LOSS FUNCTION ====================

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross-Entropy Loss
    
    Overfitting'i azaltır ve model kalibrasyonunu iyileştirir.
    
    Args:
        epsilon (float): Smoothing parameter (default: 0.1)
        reduction (str): Reduction method - 'mean', 'sum', or 'none'
    """
    
    def __init__(self, epsilon=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, preds, target):
        """
        Args:
            preds: (batch_size, num_classes) - Model logits
            target: (batch_size,) - Ground truth labels
        
        Returns:
            loss: Scalar loss value
        """
        num_classes = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        
        # Label smoothing
        # True label: 1 - epsilon + epsilon/K
        # Other labels: epsilon/K
        targets = torch.zeros_like(log_preds).scatter_(
            1, target.unsqueeze(1), 1
        )
        targets = (1 - self.epsilon) * targets + self.epsilon / num_classes
        
        loss = (-targets * log_preds).sum(dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ==================== OPTIMIZER & SCHEDULER ====================

def create_optimizer(model, config):
    """
    Create AdamW optimizer with differential learning rates
    
    Args:
        model: TransformerSignLanguageClassifier
        config: TransformerConfig
    
    Returns:
        optimizer: torch.optim.AdamW
    """
    # Farklı katmanlar için farklı learning rate (optional)
    param_groups = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if 'classifier' not in n and p.requires_grad],
            'lr': config.LEARNING_RATE,
            'weight_decay': config.WEIGHT_DECAY
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if 'classifier' in n and p.requires_grad],
            'lr': config.LEARNING_RATE * 10,  # Classifier için daha yüksek LR
            'weight_decay': 0  # Classifier'da weight decay yok
        }
    ]
    
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    return optimizer


def create_scheduler(optimizer, config, num_training_steps):
    """
    Create Cosine Annealing scheduler with Warmup
    
    Args:
        optimizer: torch.optim.Optimizer
        config: TransformerConfig
        num_training_steps: Total number of training steps
    
    Returns:
        scheduler: torch.optim.lr_scheduler
    """
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    
    warmup_steps = config.WARMUP_EPOCHS * num_training_steps // config.NUM_EPOCHS
    
    # Warmup phase: Linear increase
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    
    # Main phase: Cosine annealing
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps - warmup_steps,
        eta_min=config.LEARNING_RATE * 0.01
    )
    
    # Combine
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
    
    return scheduler


# ==================== HELPER FUNCTIONS ====================

def create_padding_mask(X, device):
    """
    Create padding mask for sequences
    
    Args:
        X: (batch, seq_len, features)
        device: torch device
    
    Returns:
        mask: (batch, seq_len) - True for padding positions
        None if device is MPS (workaround for MPS limitations)
    """
    # MPS doesn't support nested tensor operations in TransformerEncoder
    # Disable masking on MPS as a workaround
    if device.type == 'mps':
        return None
    
    # Eğer tüm feature'lar 0 ise padding
    mask = (X.sum(dim=-1) == 0)
    return mask


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cpu'):
    """
    Load model checkpoint and restore training state
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance to load weights into
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state
        device: Device to load checkpoint to
    
    Returns:
        start_epoch: Next epoch to continue from
        best_val_acc: Best validation accuracy so far
        best_val_f1: Best validation F1 score
        history: Training history (if available)
        patience_counter: Early stopping patience counter
    """
    print(f"\n📂 Loading checkpoint from {checkpoint_path}...")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"    Model weights loaded")
    
    # Load optimizer state (if provided)
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"    Optimizer state loaded")
    
    # Load scheduler state (if provided)
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"    Scheduler state loaded")
    
    # Get training state
    start_epoch = checkpoint.get('epoch', 0) + 1  # Next epoch
    best_val_acc = checkpoint.get('val_acc', 0.0)
    best_val_f1 = checkpoint.get('val_f1', 0.0)
    
    # Load history if available
    history = checkpoint.get('history', None)
    patience_counter = checkpoint.get('patience_counter', 0)
    
    print(f"    Resuming from epoch {start_epoch}")
    print(f"    Best val accuracy: {best_val_acc:.4f}")
    print(f"    Best val F1: {best_val_f1:.4f}")
    if history is not None:
        print(f"    Training history restored ({len(history.get('train_loss', []))} epochs)")
    print(f"    Early stopping patience counter: {patience_counter}")
    
    return start_epoch, best_val_acc, best_val_f1, history, patience_counter


def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, val_f1, config, filename, history=None, patience_counter=0):
    """
    Save model checkpoint with full training state
    
    Args:
        model: Model instance
        optimizer: Optimizer instance
        scheduler: Scheduler instance
        epoch: Current epoch
        val_acc: Validation accuracy
        val_f1: Validation F1 score
        config: Configuration object
        filename: Checkpoint filename
        history: Training history dictionary (optional)
        patience_counter: Early stopping patience counter
    
    Returns:
        filepath: Saved checkpoint path
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_acc': val_acc,
        'val_f1': val_f1,
        'config': vars(config),
        'history': history,
        'patience_counter': patience_counter
    }
    
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    filepath = os.path.join(config.CHECKPOINT_DIR, filename)
    torch.save(checkpoint, filepath)
    
    return filepath


# ==================== TRAINING & VALIDATION ====================

def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, config, epoch):
    """
    Train for one epoch
    
    Returns:
        train_loss, train_acc
    """
    model.train()
    
    total_loss = 0
    all_preds = []
    all_targets = []
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}/{config.NUM_EPOCHS} [Train]')
    
    for batch_idx, (X_batch, y_batch) in enumerate(progress_bar):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Padding mask oluştur (MPS'de None döner)
        mask = create_padding_mask(X_batch, device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(X_batch, mask=mask)
        loss = criterion(logits, y_batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
        
        optimizer.step()
        scheduler.step()
        
        # Metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(y_batch.cpu().numpy())
        
        # Update progress bar
        current_lr = optimizer.param_groups[0]['lr']
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{current_lr:.6f}'
        })
    
    # Epoch metrics
    epoch_loss = total_loss / len(dataloader)
    epoch_acc = accuracy_score(all_targets, all_preds)
    
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device, epoch, config):
    """
    Validate for one epoch
    
    Returns:
        val_loss, val_acc, val_f1
    """
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_targets = []
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}/{config.NUM_EPOCHS} [Val]  ')
    
    for X_batch, y_batch in progress_bar:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Padding mask (MPS'de None döner)
        mask = create_padding_mask(X_batch, device)
        
        # Forward pass
        logits = model(X_batch, mask=mask)
        loss = criterion(logits, y_batch)
        
        # Metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(y_batch.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}'
        })
    
    # Metrics
    val_loss = total_loss / len(dataloader)
    val_acc = accuracy_score(all_targets, all_preds)
    val_f1 = f1_score(all_targets, all_preds, average='macro')
    
    return val_loss, val_acc, val_f1


# ==================== MAIN TRAINING LOOP ====================

def main():
    """Main training function"""
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train Transformer Sign Language Classifier')
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from (e.g., checkpoints/last_model.pth)'
    )
    parser.add_argument(
        '--resume-from-best',
        action='store_true',
        help='Resume from best_model.pth checkpoint'
    )
    args = parser.parse_args()
    
    # Configuration
    config = TransformerConfig()
    
    print("=" * 80)
    print("TRANSFORMER SIGN LANGUAGE CLASSIFIER - TRAINING")
    print("=" * 80)
    
    # Device selection: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\nDevice: CUDA ({torch.cuda.get_device_name(0)})")
        print(f"   CUDA Version: {torch.version.cuda}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"\nDevice: MPS (Apple Silicon GPU)")
        print(f"   ⚡ Using Metal Performance Shaders")
    else:
        device = torch.device('cpu')
        print(f"\nDevice: CPU")
        print(f"   Training will be slow. Consider using GPU.")
    
    # Load data
    print(f"\n📂 Loading data from {config.PROCESSED_DATA_DIR}...")
    
    try:
        X_train = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'X_train.npy'))
        y_train = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'y_train.npy'))
        X_val = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'X_val.npy'))
        y_val = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'y_val.npy'))
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please run data preparation scripts first:")
        print("  1. python scripts/01_select_videos.py")
        print("  2. python scripts/02_extract_keypoints.py")
        print("  3. python scripts/03_normalize_data.py")
        return
    
    print(f"   Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Dataset statistics
    print(f"\nDataset Statistics:")
    print(f"   Train samples: {len(X_train)}")
    print(f"   Val samples: {len(X_val)}")
    print(f"   Sequence length: {X_train.shape[1]}")
    print(f"   Feature dimension: {X_train.shape[2]}")
    
    # Class distribution
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    print(f"\n   Train class distribution:")
    for idx, count in zip(unique_train, counts_train):
        # idx is 0-indexed label, map to original class_id for display
        original_class_id = config.TARGET_CLASS_IDS[idx]
        class_name = config.CLASS_NAMES[idx]
        print(f"      Label {idx} (ClassId {original_class_id}, {class_name}): {count} samples")
    
    # Create datasets
    train_dataset = SignLanguageDataset(X_train, y_train)
    val_dataset = SignLanguageDataset(X_val, y_val)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    print(f"\n   Batches per epoch: {len(train_loader)} (train), {len(val_loader)} (val)")
    
    # Create model
    print(f"\nCreating model...")
    model = TransformerSignLanguageClassifier(
        input_dim=config.INPUT_DIM,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT,
        num_classes=config.NUM_CLASSES,
        max_seq_length=config.MAX_SEQ_LENGTH,
        pooling_type=config.POOLING_TYPE
    ).to(device)
    
    print_model_summary(model, input_shape=(config.BATCH_SIZE, X_train.shape[1], X_train.shape[2]))
    
    # Loss, optimizer, scheduler
    criterion = LabelSmoothingCrossEntropy(epsilon=config.LABEL_SMOOTHING)
    optimizer = create_optimizer(model, config)
    
    num_training_steps = len(train_loader) * config.NUM_EPOCHS
    scheduler = create_scheduler(optimizer, config, num_training_steps)
    
    print(f"\nTraining Configuration:")
    print(f"   Loss: Label Smoothing Cross-Entropy (ε={config.LABEL_SMOOTHING})")
    print(f"   Optimizer: AdamW (lr={config.LEARNING_RATE}, wd={config.WEIGHT_DECAY})")
    print(f"   Scheduler: Cosine Annealing with Warmup ({config.WARMUP_EPOCHS} epochs)")
    print(f"   Gradient Clipping: {config.GRADIENT_CLIP}")
    print(f"   Early Stopping: {config.EARLY_STOPPING_PATIENCE} epochs patience")
    
    # Training state initialization
    start_epoch = 1
    best_val_acc = 0.0
    best_val_f1 = 0.0
    patience_counter = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'lr': []
    }
    
    # Resume from checkpoint if specified
    if args.resume or args.resume_from_best:
        if args.resume_from_best:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
        else:
            checkpoint_path = args.resume
        
        try:
            start_epoch, best_val_acc, best_val_f1, loaded_history, patience_counter = load_checkpoint(
                checkpoint_path, model, optimizer, scheduler, device
            )
            
            # Restore history if available
            if loaded_history is not None:
                history = loaded_history
            
            print(f"\nSuccessfully loaded checkpoint!")
            print(f"   Training will resume from epoch {start_epoch}")
            
        except FileNotFoundError as e:
            print(f"\nWarning: {e}")
            print(f"   Starting fresh training from epoch 1")
            start_epoch = 1
        except Exception as e:
            print(f"\nError loading checkpoint: {e}")
            print(f"   Starting fresh training from epoch 1")
            start_epoch = 1
    
    # Training loop
    print(f"\n{'='*80}")
    if start_epoch > 1:
        print(f"RESUMING TRAINING from Epoch {start_epoch}")
    else:
        print(f"TRAINING START")
    print(f"{'='*80}\n")
    
    start_time = datetime.now()
    
    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, config, epoch
        )
        
        # Validate
        val_loss, val_acc, val_f1 = validate_epoch(
            model, val_loader, criterion, device, epoch, config
        )
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Print epoch summary
        print(f"\n{'─'*80}")
        print(f"Epoch {epoch}/{config.NUM_EPOCHS} Summary:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1: {val_f1:.4f}")
        print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            patience_counter = 0
            
            filepath = save_checkpoint(
                model, optimizer, scheduler, epoch, val_acc, val_f1, config, 
                'best_model.pth', history, patience_counter
            )
            print(f"   Best model saved! (Val Acc: {val_acc:.4f}) → {filepath}")
        else:
            patience_counter += 1
            print(f"   No improvement ({patience_counter}/{config.EARLY_STOPPING_PATIENCE})")
        
        # Save last model
        if epoch % config.SAVE_FREQUENCY == 0:
            filepath = save_checkpoint(
                model, optimizer, scheduler, epoch, val_acc, val_f1, config, 
                'last_model.pth', history, patience_counter
            )
            print(f"   Checkpoint saved → {filepath}")
        
        # Early stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
        
        print(f"{'─'*80}\n")
    
    # Training completed
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"{'='*80}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"   Duration: {duration}")
    print(f"   Best Val Accuracy: {best_val_acc:.4f}")
    print(f"   Total Epochs: {epoch}")
    
    # Save history
    os.makedirs(config.LOG_DIR, exist_ok=True)
    history_path = os.path.join(config.LOG_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"   Training history saved: {history_path}")
    
    print(f"\nNext steps:")
    print(f"   1. Evaluate model: python evaluate.py")
    print(f"   2. Visualize attention: python visualize_attention.py")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

