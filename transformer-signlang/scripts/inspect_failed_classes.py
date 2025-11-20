"""
Başarısız Sınıfların Data Distribution Analizi
0% F1 alan sınıfları (nasil, okul, seker) detaylı inceler
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config import TransformerConfig

config = TransformerConfig()

# Failed classes
failed_classes = ['nasil', 'okul', 'seker']

print('=' * 80)
print('🔴 BAŞARISIZ SINIFLARIN DATA DISTRIBUTION ANALİZİ')
print('=' * 80)
print(f'\nİncelenen sınıflar: {", ".join(failed_classes)}')
print('─' * 80)

# Check each split
for split in ['train', 'val', 'test']:
    print(f'\n📂 {split.upper()} SET:')
    print('─' * 80)
    
    # Load labels
    y = np.load(f'data/processed/y_{split}.npy')
    X = np.load(f'data/processed/X_{split}.npy')
    
    print(f'Total samples: {len(y)}')
    print(f'Sequence shape: {X.shape}')
    print()
    
    for class_name in failed_classes:
        try:
            idx = config.CLASS_NAMES.index(class_name)
            count = (y == idx).sum()
            percentage = count / len(y) * 100
            
            print(f'   {class_name:15s}: {count:5d} samples ({percentage:5.2f}%)')
            
            # Check if any samples have all-zero features
            if count > 0:
                class_samples = X[y == idx]
                zero_frames = (class_samples.sum(axis=-1) == 0).sum(axis=-1)
                avg_zero_frames = zero_frames.mean()
                max_zero_frames = zero_frames.max()
                print(f'      → Avg zero frames: {avg_zero_frames:.1f} / {X.shape[1]}')
                print(f'      → Max zero frames: {max_zero_frames} / {X.shape[1]}')
                
        except ValueError:
            print(f'   {class_name:15s}: NOT IN CLASS_NAMES!')

print('\n' + '=' * 80)
print('🔍 ANALİZ:')
print('=' * 80)

# Check total class distribution
print('\n📊 Genel Sınıf Dağılımı:')
for split in ['train', 'val', 'test']:
    y = np.load(f'data/processed/y_{split}.npy')
    unique, counts = np.unique(y, return_counts=True)
    print(f'{split:10s}: {len(unique)} unique classes, min={counts.min()}, max={counts.max()}, avg={counts.mean():.1f}')

# Check if failed classes exist in CLASS_NAMES
print('\n🔤 CLASS_NAMES Kontrolü:')
for class_name in failed_classes:
    if class_name in config.CLASS_NAMES:
        idx = config.CLASS_NAMES.index(class_name)
        print(f'✅ {class_name:15s} → Index: {idx}')
    else:
        print(f'❌ {class_name:15s} → CLASS_NAMES listesinde YOK!')

print('\n' + '=' * 80)

