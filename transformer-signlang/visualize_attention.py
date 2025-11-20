#!/usr/bin/env python3
"""
Transformer Sign Language Classifier - Attention Visualization
--------------------------------------------------------------
Transformer attention weights'leri görselleştirme scripti

Kullanım:
    python visualize_attention.py [--checkpoint CHECKPOINT_PATH] [--num_samples N]

Gereksinimler:
    - data/processed/X_test.npy, y_test.npy
    - checkpoints/best_model.pth (veya belirtilen checkpoint)
    
Çıktılar:
    - results/attention/sample_{i}_layer_{l}_attention.png
    - results/attention/sample_{i}_avg_attention.png
    - results/attention/layer_wise_attention_stats.png
    - results/attention/head_wise_attention_stats.png
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

# Proje root'unu path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import TransformerConfig
from models.transformer_model import TransformerSignLanguageClassifier


# ==================== ATTENTION EXTRACTION ====================

class AttentionExtractor:
    """
    Extract attention weights from Transformer model
    
    This class hooks into the Transformer encoder layers to extract
    attention weights during forward pass.
    """
    
    def __init__(self, model):
        """
        Args:
            model: TransformerSignLanguageClassifier instance
        """
        self.model = model
        self.attention_weights = []
        self.hooks = []
    
    def _attention_hook(self, module, input, output):
        """
        Hook function to capture attention weights
        
        Args:
            module: TransformerEncoderLayer
            input: Input to the layer
            output: Output from the layer
        """
        # TransformerEncoderLayer'ın self_attn modülü MultiheadAttention
        # Bu modülün forward pass'inde attention weights hesaplanır
        # Ancak default olarak return edilmez, bu yüzden biz kendimiz hesaplayacağız
        pass
    
    def register_hooks(self):
        """Register forward hooks to all encoder layers"""
        for layer in self.model.transformer_encoder.layers:
            hook = layer.register_forward_hook(self._attention_hook)
            self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_attention_weights(self, x, mask=None):
        """
        Extract attention weights for input
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            mask: Padding mask (batch_size, seq_len)
        
        Returns:
            attention_weights: List of attention weights per layer
                               Each element: (batch_size, num_heads, seq_len, seq_len)
        """
        self.attention_weights = []
        
        # We need to manually extract attention weights
        # by calling the attention modules directly
        
        batch_size, seq_len, _ = x.size()
        
        # Input projection
        x = self.model.input_projection(x)  # (batch, seq_len, d_model)
        x = self.model.pos_encoder(x)
        
        # Prepare mask for Transformer (True = masked positions)
        if mask is not None:
            # Transformer expects: (batch, seq_len) where True = masked
            src_key_padding_mask = mask
        else:
            src_key_padding_mask = None
        
        # Extract attention weights layer by layer
        attention_weights = []
        
        # Transpose for Transformer: (seq_len, batch, d_model)
        x = x.transpose(0, 1)
        
        # Go through each encoder layer
        for layer_idx, layer in enumerate(self.model.transformer_encoder.layers):
            # Get self-attention module
            self_attn = layer.self_attn
            
            # We need to manually compute attention to get weights
            # self_attn is nn.MultiheadAttention
            
            # Save input for residual connection
            x_input = x
            
            # Self-attention with attention weights
            attn_output, attn_weights = self_attn(
                x, x, x,
                key_padding_mask=src_key_padding_mask,
                need_weights=True,
                average_attn_weights=False  # Get per-head attention weights
            )
            
            # attn_weights: (batch, num_heads, seq_len, seq_len)
            attention_weights.append(attn_weights.detach().cpu())
            
            # Continue with rest of the layer
            x = layer.dropout1(attn_output)
            x = layer.norm1(x + x_input)
            
            # Feedforward
            ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            x = layer.dropout2(ff_output)
            x = layer.norm2(x + x)
        
        # Transpose back: (batch, seq_len, d_model)
        x = x.transpose(0, 1)
        
        return attention_weights


def create_padding_mask(X):
    """
    Create padding mask for sequences
    
    Args:
        X: (batch, seq_len, features)
    
    Returns:
        mask: (batch, seq_len) - True for padding positions
    """
    mask = (X.sum(dim=-1) == 0)
    return mask


# ==================== VISUALIZATION FUNCTIONS ====================

def plot_attention_heatmap(attention, title, save_path, vmin=0, vmax=None):
    """
    Plot attention heatmap
    
    Args:
        attention: Attention weights (seq_len, seq_len)
        title: Plot title
        save_path: Path to save figure
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
    """
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        attention,
        cmap='YlOrRd',
        cbar_kws={'label': 'Attention Weight'},
        square=True,
        linewidths=0.1,
        linecolor='gray',
        vmin=vmin,
        vmax=vmax,
        xticklabels=10,
        yticklabels=10
    )
    
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Key Position (Frame Index)', fontsize=11, fontweight='bold')
    plt.ylabel('Query Position (Frame Index)', fontsize=11, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_multi_head_attention(attention_weights, layer_idx, save_dir, sample_idx=0):
    """
    Plot attention weights for all heads in a layer
    
    Args:
        attention_weights: (batch, num_heads, seq_len, seq_len)
        layer_idx: Layer index
        save_dir: Directory to save figures
        sample_idx: Sample index in batch
    """
    batch_size, num_heads, seq_len, _ = attention_weights.size()
    
    # Create subplots for all heads
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Layer {layer_idx} - Multi-Head Attention (Sample {sample_idx})', 
                 fontsize=16, fontweight='bold')
    
    for head_idx in range(num_heads):
        row = head_idx // 4
        col = head_idx % 4
        ax = axes[row, col]
        
        attn = attention_weights[sample_idx, head_idx].numpy()
        
        im = ax.imshow(attn, cmap='YlOrRd', aspect='auto', vmin=0)
        ax.set_title(f'Head {head_idx}', fontweight='bold')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        
        # Colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'sample_{sample_idx}_layer_{layer_idx}_multihead.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_averaged_attention(attention_weights, layer_idx, save_dir, sample_idx=0):
    """
    Plot averaged attention across all heads
    
    Args:
        attention_weights: (batch, num_heads, seq_len, seq_len)
        layer_idx: Layer index
        save_dir: Directory to save figures
        sample_idx: Sample index in batch
    """
    # Average across heads
    avg_attention = attention_weights[sample_idx].mean(dim=0).numpy()
    
    save_path = os.path.join(save_dir, f'sample_{sample_idx}_layer_{layer_idx}_avg.png')
    
    plot_attention_heatmap(
        avg_attention,
        f'Layer {layer_idx} - Average Attention (Sample {sample_idx})',
        save_path
    )
    
    return save_path


def plot_attention_rollout(attention_weights_list, save_dir, sample_idx=0):
    """
    Plot attention rollout (cumulative attention across layers)
    
    Args:
        attention_weights_list: List of attention weights per layer
        save_dir: Directory to save figures
        sample_idx: Sample index in batch
    """
    # Average attention weights across heads for each layer
    avg_attentions = []
    for attn in attention_weights_list:
        avg_attn = attn[sample_idx].mean(dim=0)  # (seq_len, seq_len)
        avg_attentions.append(avg_attn)
    
    # Compute rollout (matrix multiplication across layers)
    rollout = avg_attentions[0]
    for i in range(1, len(avg_attentions)):
        rollout = torch.matmul(rollout, avg_attentions[i])
    
    rollout = rollout.numpy()
    
    save_path = os.path.join(save_dir, f'sample_{sample_idx}_attention_rollout.png')
    
    plot_attention_heatmap(
        rollout,
        f'Attention Rollout (All Layers) - Sample {sample_idx}',
        save_path
    )
    
    return save_path


def plot_attention_statistics(all_attention_weights, config, save_dir):
    """
    Plot attention statistics across layers and heads
    
    Args:
        all_attention_weights: List of attention weights (num_samples, num_layers, batch, heads, seq, seq)
        config: TransformerConfig
        save_dir: Directory to save figures
    """
    num_layers = len(all_attention_weights[0])
    num_heads = config.NHEAD
    
    # Aggregate statistics
    layer_stats = {i: [] for i in range(num_layers)}
    head_stats = {i: [] for i in range(num_heads)}
    
    for sample_attns in all_attention_weights:
        for layer_idx, layer_attn in enumerate(sample_attns):
            # layer_attn: (batch, heads, seq, seq)
            for batch_idx in range(layer_attn.size(0)):
                for head_idx in range(num_heads):
                    attn = layer_attn[batch_idx, head_idx]
                    
                    # Statistics: mean, max, entropy
                    mean_attn = attn.mean().item()
                    max_attn = attn.max().item()
                    
                    layer_stats[layer_idx].append(mean_attn)
                    head_stats[head_idx].append(mean_attn)
    
    # Plot 1: Layer-wise statistics
    fig, ax = plt.subplots(figsize=(10, 6))
    
    layer_means = [np.mean(layer_stats[i]) for i in range(num_layers)]
    layer_stds = [np.std(layer_stats[i]) for i in range(num_layers)]
    
    x = np.arange(num_layers)
    ax.bar(x, layer_means, yerr=layer_stds, capsize=5, alpha=0.7, color='steelblue')
    ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Attention Weight', fontsize=12, fontweight='bold')
    ax.set_title('Layer-wise Attention Statistics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'layer_wise_attention_stats.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {save_path}")
    
    # Plot 2: Head-wise statistics
    fig, ax = plt.subplots(figsize=(10, 6))
    
    head_means = [np.mean(head_stats[i]) for i in range(num_heads)]
    head_stds = [np.std(head_stats[i]) for i in range(num_heads)]
    
    x = np.arange(num_heads)
    ax.bar(x, head_means, yerr=head_stds, capsize=5, alpha=0.7, color='coral')
    ax.set_xlabel('Head Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Attention Weight', fontsize=12, fontweight='bold')
    ax.set_title('Head-wise Attention Statistics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'head_wise_attention_stats.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {save_path}")


# ==================== MAIN ====================

def main():
    """Main visualization function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Visualize Transformer Attention Weights')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (default: checkpoints/best_model.pth)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='Number of test samples to visualize (default: 5)'
    )
    args = parser.parse_args()
    
    # Configuration
    config = TransformerConfig()
    
    print("=" * 80)
    print("🎨 TRANSFORMER ATTENTION VISUALIZATION")
    print("=" * 80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    # Load test data
    print(f"\n📂 Loading test data from {config.PROCESSED_DATA_DIR}...")
    
    try:
        X_test = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'X_test.npy'))
        y_test = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'y_test.npy'))
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please run data preparation scripts first.")
        return
    
    print(f"   ✅ Test data: {X_test.shape}")
    
    # Load model
    checkpoint_path = args.checkpoint if args.checkpoint else os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
    
    print(f"\n🏗️  Loading model from {checkpoint_path}...")
    
    if not os.path.exists(checkpoint_path):
        print(f"\n❌ Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first: python train.py")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
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
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"   ✅ Model loaded successfully!")
    
    # Create attention extractor
    extractor = AttentionExtractor(model)
    
    # Create output directory
    attention_dir = os.path.join(config.RESULTS_DIR, 'attention')
    os.makedirs(attention_dir, exist_ok=True)
    
    # Select random samples
    num_samples = min(args.num_samples, len(X_test))
    sample_indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    print(f"\n🎨 Visualizing attention for {num_samples} samples...")
    print(f"   Sample indices: {sample_indices.tolist()}")
    
    all_attention_weights = []
    
    for i, sample_idx in enumerate(tqdm(sample_indices, desc='Extracting attention')):
        # Get sample
        x = torch.FloatTensor(X_test[sample_idx:sample_idx+1]).to(device)  # (1, seq_len, 258)
        y_true = y_test[sample_idx]
        
        # Create mask (disable for visualization to avoid compatibility issues)
        # mask = create_padding_mask(x)
        mask = None
        
        # Extract attention weights
        with torch.no_grad():
            attention_weights = extractor.get_attention_weights(x, mask)
        
        all_attention_weights.append(attention_weights)
        
        # Get prediction
        with torch.no_grad():
            logits = model(x, mask)
            y_pred = logits.argmax(dim=1).item()
        
        # y_true and y_pred are already 0-indexed (0, 1, 2)
        true_class = config.CLASS_NAMES[y_true]
        pred_class = config.CLASS_NAMES[y_pred]
        
        print(f"\n   Sample {i} (Index: {sample_idx}):")
        print(f"      True: {true_class} | Predicted: {pred_class}")
        
        # Visualize each layer
        for layer_idx, layer_attn in enumerate(attention_weights):
            # Multi-head attention
            save_path = plot_multi_head_attention(layer_attn, layer_idx, attention_dir, sample_idx=i)
            
            # Average attention
            save_path = plot_averaged_attention(layer_attn, layer_idx, attention_dir, sample_idx=i)
        
        # Attention rollout
        save_path = plot_attention_rollout(attention_weights, attention_dir, sample_idx=i)
    
    # Attention statistics
    print(f"\n📊 Computing attention statistics...")
    plot_attention_statistics(all_attention_weights, config, attention_dir)
    
    print(f"\n{'='*80}")
    print(f"✅ VISUALIZATION COMPLETED")
    print(f"{'='*80}")
    print(f"\n📁 Visualizations saved to: {attention_dir}/")
    print(f"   - Per-sample multi-head attention heatmaps")
    print(f"   - Per-layer averaged attention heatmaps")
    print(f"   - Attention rollout visualizations")
    print(f"   - Layer-wise and head-wise statistics")
    print(f"\n   Total files: ~{num_samples * (config.NUM_ENCODER_LAYERS * 2 + 1) + 2}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

