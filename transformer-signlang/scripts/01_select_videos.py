#!/usr/bin/env python3
"""
Script 01: Video Seçimi (Train/Val/Test)
-----------------------------------------
İlk 3 kelimeye (acele, acikmak, agac) ait videoları Train, Validation ve Test 
setlerinden ayrı ayrı seçer ve CSV'lere kaydeder.

Kullanım:
    python scripts/01_select_videos.py

Çıktı:
    data/selected_videos_train.csv - Train videoları
    data/selected_videos_val.csv - Validation videoları
    data/selected_videos_test.csv - Test videoları
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Proje root'unu path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import TransformerConfig


def process_split(labels_path, video_dir, target_classes, class_names, split_name):
    """Bir split için video seçimi yapar"""
    
    print(f"\n{'='*70}")
    print(f"🎯 {split_name.upper()} SET İŞLENİYOR")
    print(f"{'='*70}")
    
    if not os.path.exists(labels_path):
        print(f"   ⚠️  Etiket dosyası bulunamadı: {labels_path}")
        return None
    
    # CSV'yi oku
    labels_df = pd.read_csv(labels_path, header=None, names=['video_id', 'class_id'])
    
    print(f"   ✅ Toplam {len(labels_df)} video etiketi yüklendi")
    
    # Hedef sınıflara ait videoları filtrele
    filtered_df = labels_df[labels_df['class_id'].isin(target_classes)].copy()
    
    print(f"   ✅ Filtrelenmiş video sayısı: {len(filtered_df)}")
    
    # Sınıf dağılımı
    print(f"\n   📊 Sınıf Dağılımı:")
    for class_id in target_classes:
        count = (filtered_df['class_id'] == class_id).sum()
        class_name = class_names[target_classes.index(class_id)]
        print(f"      ClassId {class_id} ({class_name}): {count} video")
    
    # Video yollarını oluştur
    video_paths = []
    missing_videos = []
    
    for idx, row in filtered_df.iterrows():
        video_id = row['video_id']
        class_id = row['class_id']
        
        # Color video yolu
        color_path = os.path.join(video_dir, f'{video_id}_color.mp4')
        
        # Video var mı kontrol et
        if os.path.exists(color_path):
            video_paths.append({
                'video_id': video_id,
                'class_id': class_id,
                'path': color_path,
                'split': split_name
            })
        else:
            missing_videos.append(video_id)
    
    print(f"\n   ✅ Bulunan videolar: {len(video_paths)}")
    
    if missing_videos:
        print(f"   ⚠️  Bulunamayan videolar: {len(missing_videos)}")
        if len(missing_videos) <= 5:
            for vid in missing_videos[:5]:
                print(f"      - {vid}")
    
    return pd.DataFrame(video_paths)


def main():
    """Ana fonksiyon"""
    config = TransformerConfig()
    
    print("=" * 80)
    print("📹 VİDEO SEÇİMİ - TRAIN/VAL/TEST SETLER")
    print("=" * 80)
    
    # Hedef sınıflar
    target_classes = config.TARGET_CLASS_IDS  # 10 kelime
    class_names = config.CLASS_NAMES  # 10 kelime
    
    print(f"\n🎯 Hedef Sınıflar:")
    for class_id, class_name in zip(target_classes, class_names):
        print(f"   ClassId {class_id}: {class_name}")
    
    # Çıktı dizinini oluştur
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    # TRAIN SET
    train_df = process_split(
        labels_path=config.TRAIN_LABELS_CSV,
        video_dir=config.TRAIN_VIDEO_DIR,
        target_classes=target_classes,
        class_names=class_names,
        split_name='train'
    )
    
    # VALIDATION SET
    val_labels_path = os.path.join(config.BASE_DATA_DIR, 'Test Data & Valid, Labels/ground_truth 2.csv')
    val_video_dir = os.path.join(config.BASE_DATA_DIR, 'Validation Data/val')
    
    val_df = process_split(
        labels_path=val_labels_path,
        video_dir=val_video_dir,
        target_classes=target_classes,
        class_names=class_names,
        split_name='val'
    )
    
    # TEST SET
    test_labels_path = os.path.join(config.BASE_DATA_DIR, 'Test Data & Valid, Labels/ground_truth.csv')
    test_video_dir = os.path.join(config.BASE_DATA_DIR, 'Test Data & Valid, Labels/test')
    
    test_df = process_split(
        labels_path=test_labels_path,
        video_dir=test_video_dir,
        target_classes=target_classes,
        class_names=class_names,
        split_name='test'
    )
    
    # Kaydet
    print(f"\n{'='*80}")
    print("💾 SONUÇLAR KAYDEDİLİYOR")
    print(f"{'='*80}")
    
    if train_df is not None:
        train_path = os.path.join(config.DATA_DIR, 'selected_videos_train.csv')
        train_df.to_csv(train_path, index=False)
        print(f"   ✅ Train: {train_path} ({len(train_df)} video)")
    
    if val_df is not None:
        val_path = os.path.join(config.DATA_DIR, 'selected_videos_val.csv')
        val_df.to_csv(val_path, index=False)
        print(f"   ✅ Val:   {val_path} ({len(val_df)} video)")
    
    if test_df is not None:
        test_path = os.path.join(config.DATA_DIR, 'selected_videos_test.csv')
        test_df.to_csv(test_path, index=False)
        print(f"   ✅ Test:  {test_path} ({len(test_df)} video)")
    
    # Özet
    print(f"\n{'='*80}")
    print("📊 GENEL ÖZET")
    print(f"{'='*80}")
    
    total_train = len(train_df) if train_df is not None else 0
    total_val = len(val_df) if val_df is not None else 0
    total_test = len(test_df) if test_df is not None else 0
    total_all = total_train + total_val + total_test
    
    print(f"\n   Train: {total_train:3d} videos ({total_train/total_all*100:5.1f}%)" if total_all > 0 else "")
    print(f"   Val:   {total_val:3d} videos ({total_val/total_all*100:5.1f}%)" if total_all > 0 else "")
    print(f"   Test:  {total_test:3d} videos ({total_test/total_all*100:5.1f}%)" if total_all > 0 else "")
    print(f"   {'─'*40}")
    print(f"   Total: {total_all:3d} videos")
    
    print(f"\n{'='*80}")
    print(f"✅ Video seçimi tamamlandı!")
    print(f"📌 Sıradaki adım: Keypoint çıkarımı (02_extract_keypoints.py)")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

