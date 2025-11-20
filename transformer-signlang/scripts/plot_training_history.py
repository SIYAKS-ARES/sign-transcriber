"""
Training History Visualization
Eğitim sürecini görselleştirir
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import matplotlib.pyplot as plt
import numpy as np

# Load history
with open('logs/training_history.json') as f:
    history = json.load(f)

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Loss curves
axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
axes[0, 0].set_title('Loss vs Epoch', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Accuracy curves
axes[0, 1].plot(history['train_acc'], label='Train Acc', linewidth=2)
axes[0, 1].plot(history['val_acc'], label='Val Acc', linewidth=2)
axes[0, 1].set_title('Accuracy vs Epoch', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='100%')

# F1 score
axes[1, 0].plot(history['val_f1'], label='Val F1', color='green', linewidth=2)
axes[1, 0].set_title('Val F1 Score vs Epoch', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('F1 Score')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Learning Rate
axes[1, 1].plot(history['lr'], label='Learning Rate', color='orange', linewidth=2)
axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('LR')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_yscale('log')

plt.tight_layout()
plt.savefig('results/training_curves.png', dpi=150, bbox_inches='tight')
print('✅ Saved: results/training_curves.png')

# Print summary
print('\n📊 Training Summary:')
print('─' * 60)
print(f'Total Epochs: {len(history["train_loss"])}')
print(f'Best Train Acc: {max(history["train_acc"]):.4f} (Epoch {np.argmax(history["train_acc"]) + 1})')
print(f'Best Val Acc: {max(history["val_acc"]):.4f} (Epoch {np.argmax(history["val_acc"]) + 1})')
print(f'Best Val F1: {max(history["val_f1"]):.4f} (Epoch {np.argmax(history["val_f1"]) + 1})')
print(f'Final Train Acc: {history["train_acc"][-1]:.4f}')
print(f'Final Val Acc: {history["val_acc"][-1]:.4f}')
print(f'Train-Val Gap (final): {history["train_acc"][-1] - history["val_acc"][-1]:.4f}')

