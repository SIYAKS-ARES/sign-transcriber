#!/usr/bin/env python3
"""
Transformer Sign Language Classifier - Evaluation Script
--------------------------------------------------------
Test seti değerlendirme scripti: Metrics, confusion matrix, visualizations

Kullanım:
    python evaluate.py [--checkpoint CHECKPOINT_PATH]

Gereksinimler:
    - data/processed/X_test.npy, y_test.npy
    - checkpoints/best_model.pth (veya belirtilen checkpoint)
    
Çıktılar:
    - results/evaluation_report.json
    - results/confusion_matrix_raw.csv
    - results/confusion_matrix_normalized.csv
    - results/confusion_matrix_raw.png
    - results/confusion_matrix_normalized.png
    - results/per_class_metrics.csv
    - results/per_class_metrics.png
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import json

# Proje root'unu path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import TransformerConfig
from models.transformer_model import TransformerSignLanguageClassifier


# ==================== EVALUATION FUNCTIONS ====================

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


@torch.no_grad()
def evaluate_model(model, X_test, y_test, device, batch_size=32):
    """
    Evaluate model on test set
    
    Args:
        model: TransformerSignLanguageClassifier
        X_test: Test features (N, seq_len, 258)
        y_test: Test labels (N,)
        device: torch.device
        batch_size: Batch size for evaluation
    
    Returns:
        all_preds: Predicted labels (N,)
        all_probs: Prediction probabilities (N, num_classes)
        all_targets: Ground truth labels (N,)
    """
    model.eval()
    
    all_preds = []
    all_probs = []
    all_targets = []
    
    num_samples = len(X_test)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print(f"\n🔍 Evaluating on {num_samples} test samples...")
    
    for i in tqdm(range(num_batches), desc='Evaluation'):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        
        # Get batch
        X_batch = X_test[start_idx:end_idx]
        y_batch = y_test[start_idx:end_idx]
        
        # Convert to tensor
        if isinstance(X_batch, np.ndarray):
            X_batch = torch.FloatTensor(X_batch)
        if isinstance(y_batch, np.ndarray):
            y_batch = torch.LongTensor(y_batch)
        
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Padding mask
        mask = create_padding_mask(X_batch)
        
        # Forward pass
        logits = model(X_batch, mask=mask)
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        
        # Store results
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_targets.append(y_batch.cpu().numpy())
    
    # Concatenate
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    
    return all_preds, all_probs, all_targets


def compute_metrics(y_true, y_pred, class_names):
    """
    Compute evaluation metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
    
    Returns:
        metrics: Dictionary of metrics
    """
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Confusion matrix
    cm_raw = confusion_matrix(y_true, y_pred)
    cm_normalized = confusion_matrix(y_true, y_pred, normalize='true')
    
    # Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    metrics = {
        'overall': {
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
        },
        'per_class': {},
        'confusion_matrix': {
            'raw': cm_raw.tolist(),
            'normalized': cm_normalized.tolist()
        },
        'classification_report': report
    }
    
    # Per-class metrics
    for i, class_name in enumerate(class_names):
        metrics['per_class'][class_name] = {
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1_score': float(f1_per_class[i]),
            'support': int(report[class_name]['support'])
        }
    
    return metrics


# ==================== VISUALIZATION ====================

def plot_confusion_matrix(cm, class_names, title, save_path, normalize=False):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix (num_classes, num_classes)
        class_names: List of class names
        title: Plot title
        save_path: Path to save figure
        normalize: Whether to normalize
    """
    plt.figure(figsize=(10, 8))
    
    # Format
    if normalize:
        fmt = '.2f'
        cmap = 'Blues'
    else:
        fmt = 'd'
        cmap = 'Blues'
    
    # Plot
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Count' if normalize else 'Count'},
        square=True,
        linewidths=0.5,
        linecolor='gray'
    )
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}")
    plt.close()


def plot_per_class_metrics(metrics, class_names, save_path):
    """
    Plot per-class metrics (precision, recall, F1)
    
    Args:
        metrics: Metrics dictionary
        class_names: List of class names
        save_path: Path to save figure
    """
    # Extract data
    precision = [metrics['per_class'][name]['precision'] for name in class_names]
    recall = [metrics['per_class'][name]['recall'] for name in class_names]
    f1 = [metrics['per_class'][name]['f1_score'] for name in class_names]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(class_names))
    width = 0.25
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71', alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c', alpha=0.8)
    
    # Labels
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(fontsize=10, loc='lower right')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8
            )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}")
    plt.close()


def plot_prediction_confidence(probs, preds, targets, class_names, save_path):
    """
    Plot prediction confidence distribution
    
    Args:
        probs: Prediction probabilities (N, num_classes)
        preds: Predicted labels (N,)
        targets: True labels (N,)
        class_names: List of class names
        save_path: Path to save figure
    """
    # Get confidence scores (max probability)
    confidences = np.max(probs, axis=1)
    
    # Separate correct and incorrect predictions
    correct_mask = (preds == targets)
    conf_correct = confidences[correct_mask]
    conf_incorrect = confidences[~correct_mask]
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(conf_correct, bins=30, alpha=0.7, label='Correct', color='green', edgecolor='black')
    ax1.hist(conf_incorrect, bins=30, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
    ax1.set_xlabel('Prediction Confidence', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3, linestyle='--')
    
    # Box plot per class
    conf_per_class = []
    for i, class_name in enumerate(class_names):
        class_mask = (targets == i)
        conf_per_class.append(confidences[class_mask])
    
    bp = ax2.boxplot(conf_per_class, labels=class_names, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#3498db')
        patch.set_alpha(0.6)
    
    ax2.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Prediction Confidence', fontsize=12, fontweight='bold')
    ax2.set_title('Confidence by Class', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}")
    plt.close()


# ==================== SAVE RESULTS ====================

def save_results(metrics, config, save_dir):
    """
    Save evaluation results to files
    
    Args:
        metrics: Metrics dictionary
        config: TransformerConfig
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. JSON report
    report_path = os.path.join(save_dir, 'evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"   ✅ Saved: {report_path}")
    
    # 2. Confusion matrix - Raw
    cm_raw = np.array(metrics['confusion_matrix']['raw'])
    cm_raw_df = pd.DataFrame(cm_raw, index=config.CLASS_NAMES, columns=config.CLASS_NAMES)
    cm_raw_path = os.path.join(save_dir, 'confusion_matrix_raw.csv')
    cm_raw_df.to_csv(cm_raw_path)
    print(f"   ✅ Saved: {cm_raw_path}")
    
    # 3. Confusion matrix - Normalized
    cm_norm = np.array(metrics['confusion_matrix']['normalized'])
    cm_norm_df = pd.DataFrame(cm_norm, index=config.CLASS_NAMES, columns=config.CLASS_NAMES)
    cm_norm_path = os.path.join(save_dir, 'confusion_matrix_normalized.csv')
    cm_norm_df.to_csv(cm_norm_path)
    print(f"   ✅ Saved: {cm_norm_path}")
    
    # 4. Per-class metrics
    per_class_data = []
    for class_name in config.CLASS_NAMES:
        per_class_data.append({
            'class': class_name,
            'precision': metrics['per_class'][class_name]['precision'],
            'recall': metrics['per_class'][class_name]['recall'],
            'f1_score': metrics['per_class'][class_name]['f1_score'],
            'support': metrics['per_class'][class_name]['support']
        })
    
    per_class_df = pd.DataFrame(per_class_data)
    per_class_path = os.path.join(save_dir, 'per_class_metrics.csv')
    per_class_df.to_csv(per_class_path, index=False)
    print(f"   ✅ Saved: {per_class_path}")


# ==================== MAIN ====================

def main():
    """Main evaluation function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate Transformer Sign Language Classifier')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (default: checkpoints/best_model.pth)'
    )
    args = parser.parse_args()
    
    # Configuration
    config = TransformerConfig()
    
    print("=" * 80)
    print("📊 TRANSFORMER SIGN LANGUAGE CLASSIFIER - EVALUATION")
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
    print(f"   Test samples: {len(X_test)}")
    
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
    
    print(f"   ✅ Model loaded successfully!")
    print(f"   Checkpoint epoch: {checkpoint['epoch']}")
    print(f"   Validation accuracy: {checkpoint['val_acc']:.4f}")
    print(f"   Validation F1: {checkpoint['val_f1']:.4f}")
    
    # Evaluate
    all_preds, all_probs, all_targets = evaluate_model(
        model, X_test, y_test, device, batch_size=config.BATCH_SIZE
    )
    
    # Compute metrics
    print(f"\n📈 Computing metrics...")
    metrics = compute_metrics(all_targets, all_preds, config.CLASS_NAMES)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"📊 EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"\n🎯 Overall Metrics:")
    print(f"   Accuracy:          {metrics['overall']['accuracy']:.4f}")
    print(f"   Precision (macro): {metrics['overall']['precision_macro']:.4f}")
    print(f"   Recall (macro):    {metrics['overall']['recall_macro']:.4f}")
    print(f"   F1-Score (macro):  {metrics['overall']['f1_macro']:.4f}")
    print(f"\n   Precision (weighted): {metrics['overall']['precision_weighted']:.4f}")
    print(f"   Recall (weighted):    {metrics['overall']['recall_weighted']:.4f}")
    print(f"   F1-Score (weighted):  {metrics['overall']['f1_weighted']:.4f}")
    
    print(f"\n📋 Per-Class Metrics:")
    for class_name in config.CLASS_NAMES:
        cm = metrics['per_class'][class_name]
        print(f"   {class_name:10s} - Precision: {cm['precision']:.4f} | Recall: {cm['recall']:.4f} | F1: {cm['f1_score']:.4f} | Support: {cm['support']}")
    
    # Save results
    print(f"\n💾 Saving results to {config.RESULTS_DIR}...")
    save_results(metrics, config, config.RESULTS_DIR)
    
    # Visualizations
    print(f"\n🎨 Creating visualizations...")
    
    # Confusion matrices
    cm_raw = np.array(metrics['confusion_matrix']['raw'])
    cm_norm = np.array(metrics['confusion_matrix']['normalized'])
    
    plot_confusion_matrix(
        cm_raw, config.CLASS_NAMES,
        'Confusion Matrix (Raw Counts)',
        os.path.join(config.RESULTS_DIR, 'confusion_matrix_raw.png'),
        normalize=False
    )
    
    plot_confusion_matrix(
        cm_norm, config.CLASS_NAMES,
        'Confusion Matrix (Normalized)',
        os.path.join(config.RESULTS_DIR, 'confusion_matrix_normalized.png'),
        normalize=True
    )
    
    # Per-class metrics
    plot_per_class_metrics(
        metrics, config.CLASS_NAMES,
        os.path.join(config.RESULTS_DIR, 'per_class_metrics.png')
    )
    
    # Prediction confidence
    plot_prediction_confidence(
        all_probs, all_preds, all_targets, config.CLASS_NAMES,
        os.path.join(config.RESULTS_DIR, 'prediction_confidence.png')
    )
    
    print(f"\n{'='*80}")
    print(f"✅ EVALUATION COMPLETED")
    print(f"{'='*80}")
    print(f"\n📁 Results saved to: {config.RESULTS_DIR}/")
    print(f"   - evaluation_report.json")
    print(f"   - confusion_matrix_raw.csv/png")
    print(f"   - confusion_matrix_normalized.csv/png")
    print(f"   - per_class_metrics.csv/png")
    print(f"   - prediction_confidence.png")
    
    print(f"\n📌 Next step:")
    print(f"   Visualize attention: python visualize_attention.py")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

